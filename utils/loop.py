# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loop utilities."""

import jax
import jax.numpy as jnp


def _while_loop_scan_limited_cond(cond_fun, body_fun, init_val, maxiter, steps_before_next_cond):
  _body_fun = lambda _, val: body_fun(val)

  def _iter(val, steps):
    next_val = jax.lax.fori_loop(0, steps, _body_fun, val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, inner_steps):
    val, cond = tup
    next_val, next_cond = jax.lax.cond(cond, _iter, lambda x, _: (x, False), val, inner_steps)
    return (next_val, next_cond), inner_steps

  init = (init_val, cond_fun(init_val))
  return jax.lax.scan(_fun, init, steps_before_next_cond)[0][0]


def _while_loop_scan(cond_fun, body_fun, init_val, maxiter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    next_val, next_cond = jax.lax.cond(cond, _iter, lambda x: (x, False), val)
    return (next_val, next_cond), it

  init = (init_val, cond_fun(init_val))
  return jax.lax.scan(_fun, init, None, length=maxiter)[0][0]


def _while_loop_python(cond_fun, body_fun, init_val, maxiter):
  """Python based implementation (no jit, reverse-mode autodiff ok)."""
  val = init_val
  for _ in range(maxiter):
    cond = cond_fun(val)
    if not cond:
      # When condition is met, break (not jittable).
      break
    val = body_fun(val)
  return val


def _while_loop_lax(cond_fun, body_fun, init_val, maxiter):
  """lax.while_loop based implementation (jit by default, no reverse-mode)."""
  def _cond_fun(_val):
    it, val = _val
    return jnp.logical_and(cond_fun(val), it <= maxiter - 1)

  def _body_fun(_val):
    it, val = _val
    val = body_fun(val)
    return it+1, val

  return jax.lax.while_loop(_cond_fun, _body_fun, (0, init_val))[1]


def while_loop(cond_fun, body_fun, init_val, maxiter, unroll=False, jit=False, cond_exp_base=1.0):
  """A while loop with a bounded number of iterations."""

  assert cond_exp_base >= 1.0 and cond_exp_base <= 2.0

  if cond_exp_base > 1.0:
    assert unroll and jit
    # see formula from https://en.m.wikipedia.org/wiki/Geometric_series
    num_cond_evals = jnp.ceil(jnp.log((cond_exp_base - 1) * maxiter - 1) / jnp.log(cond_exp_base)) + 1
    steps_before_next_cond = cond_exp_base ** jnp.arange(num_cond_evals)
    steps_before_next_cond = steps_before_next_cond.astype(int)
    mask = (jnp.cumsum(steps_before_next_cond) >= maxiter)
    steps_before_next_cond = jnp.delete(steps_before_next_cond, jnp.where(mask)[0])
    steps_before_next_cond = jnp.append(steps_before_next_cond, maxiter - jnp.sum(steps_before_next_cond))

    def _closure_while_loop_scan_limited_cond(cond_fun, body_fun, init_val, maxiter):
      return _while_loop_scan_limited_cond(cond_fun, body_fun, init_val, maxiter, steps_before_next_cond)

    fun = _closure_while_loop_scan_limited_cond
  elif unroll:
    if jit:
      fun = _while_loop_scan
    else:
      fun = _while_loop_python
  else:
    if jit:
      fun = _while_loop_lax
    else:
      raise ValueError("unroll=False and jit=False cannot be used together")

  if jit and fun is not _while_loop_lax:
    # jit of a lax while_loop is redundant, and this jit would only
    # constrain maxiter to be static where it is not required.
    fun = jax.jit(fun, static_argnums=(0, 1, 3))

  return fun(cond_fun, body_fun, init_val, maxiter)
