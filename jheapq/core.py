"""
JAX-based Implementation of heapq.
This is an almost straightforward translation of python's native heapq implemented below:
https://github.com/python/cpython/blob/main/Lib/heapq.py

Author: @yonetaniryo

"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, Scalar


@jax.jit
def _siftup(heap: Array, pos: int, endpos: int) -> Array:
    # endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1

    class Carry(NamedTuple):
        heap: Array
        pos: int
        childpos: int

    def cond(carry):
        return carry.childpos < endpos

    def body(carry):
        rightpos = carry.childpos + 1
        cond = (rightpos < endpos) & ~(
            carry.heap[carry.childpos] < carry.heap[rightpos]
        )
        childpos = jax.lax.cond(
            cond, lambda _: rightpos, lambda _: carry.childpos, None
        )
        heap = carry.heap.at[carry.pos].set(carry.heap[childpos])
        pos = childpos
        childpos = 2 * pos + 1
        carry = Carry(heap=heap, pos=pos, childpos=childpos)
        return carry

    carry = Carry(heap=heap, pos=pos, childpos=childpos)
    carry = jax.lax.while_loop(cond, body, carry)
    heap = carry.heap
    heap = heap.at[carry.pos].set(newitem)
    heap = _siftdown(heap, startpos, carry.pos)
    return heap


@jax.jit
def _siftdown(heap: Array, startpos: int, pos: int) -> Array:
    newitem = heap[pos]

    class Carry(NamedTuple):
        heap: Array
        pos: int
        continue_flag: bool

    def cond(carry):
        return (carry.pos > startpos) & carry.continue_flag

    def body(carry):
        parentpos = (carry.pos - 1) >> 1
        parent = carry.heap[parentpos]
        cond = newitem < parent
        heap = carry.heap.at[carry.pos].set(parent)
        heap = jax.lax.cond(cond, lambda _: heap, lambda _: carry.heap, None)
        pos = jax.lax.cond(cond, lambda _: parentpos, lambda _: carry.pos, None)
        carry = Carry(heap=heap, pos=pos, continue_flag=cond)

        return carry

    carry = Carry(heap=heap, pos=pos, continue_flag=True)
    carry = jax.lax.while_loop(cond, body, carry)
    heap = carry.heap
    heap = heap.at[carry.pos].set(newitem)

    return heap


def heapify_fixedsize(data: Array) -> Array:
    size = len(data)
    return _heapify(data, size)


def heapify(data: Array, max_size: int = 1024) -> Array:
    size = len(data)
    assert size <= max_size, "max_size should be greater than or equal to len(data)"
    data_extended = jnp.hstack((data, jnp.ones(max_size - size) * jnp.inf))
    return _heapify(data_extended, size)[:size]


@jax.jit
def _heapify(data: Array, size: int) -> Array:
    def for_body(i, heap):
        i_rev = size // 2 - i - 1
        heap = _siftup(heap, i_rev, size)
        return heap

    return jax.lax.fori_loop(0, size // 2, for_body, data)


@jax.jit
def heappush_fixedsize(data: Array, item: Scalar) -> Array:
    data = jnp.hstack((data, jnp.array(item)))
    return _siftdown(data, 0, len(data) - 1)


@jax.jit
def heappop_fixedsize(data: Array) -> tuple[Scalar, Array]:
    lastelt = data[-1]
    ret = data[0]
    data = data.at[0].set(lastelt)
    data = _siftup(data, 0, len(data))[:-1]
    return ret, data


def heappush(data: Array, item: Scalar, max_size: int = 1024) -> Array:
    size = len(data)
    assert (
        size + 1 <= max_size
    ), "max_size should be greater than or equal to len(data) + 1"
    data = jnp.hstack((data, jnp.array(item)))
    data_extended = jnp.hstack((data, jnp.ones(max_size - size) * jnp.inf))
    return _siftdown(data_extended, 0, size - 1)[: size + 1]


def heappop(data: Array, max_size: int = 1024) -> tuple[Scalar, Array]:
    lastelt = data[-1]
    ret = data[0]
    data = data.at[0].set(lastelt)
    size = len(data)
    data_extended = jnp.hstack((data, jnp.ones(max_size - size) * jnp.inf))
    data = _siftup(data_extended, 0, len(data))[: size - 1]
    return ret, data
