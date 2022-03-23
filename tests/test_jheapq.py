import heapq

import jax
import jax.numpy as jnp
import pytest
from jheapq import heapify, heappop, heappush


@pytest.fixture
def setup():
    data = jax.random.categorical(
        jax.random.PRNGKey(0), jnp.ones(150), shape=(101,)
    ).astype(float)
    data_list = data.tolist()

    return data, data_list


def test_heapify(setup):
    data, data_list = setup
    heapified = heapify(data)

    heapq.heapify(data_list)
    heapified_native = jnp.array(data_list)

    jnp.allclose(heapified, heapified_native)


def test_heappop(setup):
    data, data_list = setup
    heapified = heapify(data)
    ret, heapified = heappop(heapified)

    heapq.heapify(data_list)
    ret_native = heapq.heappop(data_list)
    heapified_native = jnp.array(data_list)

    jnp.allclose(heapified, heapified_native)
    jnp.allclose(ret, ret_native)


def test_heappush(setup):
    data, data_list = setup
    heapified = heapify(data)
    heapified = heappush(1000, heapified)

    heapq.heapify(data_list)
    heapq.heappush(data_list, 1000)
    heapified_native = jnp.array(data_list)

    jnp.allclose(heapified, heapified_native)
