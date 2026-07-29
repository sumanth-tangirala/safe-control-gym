'''Attributes the codebase reads through a wrapper must still resolve.

gymnasium 1.0 removed Wrapper.__getattr__ passthrough. These are the attributes
that call sites actually use; see the FORWARDED tuple in each wrapper.
'''
import numpy as np
import pytest

from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.experiments.base_experiment import RecordDataWrapper
from safe_control_gym.utils.registration import make

ATTRS = ['GUI', 'CTRL_FREQ', 'constraints', 'done_on_out_of_bound',
         'symbolic', 'state']


@pytest.mark.parametrize('wrapper_cls', [RecordDataWrapper, RecordEpisodeStatistics])
@pytest.mark.parametrize('attr', ATTRS)
def test_attribute_forwards(wrapper_cls, attr):
    env = make('cartpole')
    wrapped = wrapper_cls(env)
    wrapped.reset()
    expected = getattr(env, attr)          # raises if the env itself lacks it
    actual = getattr(wrapped, attr)        # raises if forwarding is missing
    assert actual is expected or np.array_equal(actual, expected)
    env.close()


@pytest.mark.parametrize('wrapper_cls', [RecordDataWrapper, RecordEpisodeStatistics])
def test_unknown_attribute_still_raises(wrapper_cls):
    '''Allowlisted forwarding, not blanket -- a typo must not resolve.'''
    env = make('cartpole')
    wrapped = wrapper_cls(env)
    with pytest.raises(AttributeError):
        wrapped.definitely_not_an_attribute
    env.close()
