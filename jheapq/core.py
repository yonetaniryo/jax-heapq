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


def _siftup(heap: Array, pos: int) -> Array:
    endpos = len(heap)
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


@jax.jit
def heapify(heap: Array) -> Array:
    n = len(heap)

    def for_body(i, heap):
        i_rev = n // 2 - i - 1
        heap = _siftup(heap, i_rev)
        return heap

    return jax.lax.fori_loop(0, n // 2, for_body, heap)


@jax.jit
def heappush(heap: Array, item: Scalar) -> Array:
    heap = jnp.hstack((heap, jnp.array(item)))
    return _siftdown(heap, 0, len(heap) - 1)


@jax.jit
def heappop(heap: Array) -> tuple[Scalar, Array]:
    lastelt = heap[-1]
    ret = heap[0]
    heap = heap.at[0].set(lastelt)
    heap = _siftup(heap, 0)[:-1]
    return ret, heap
