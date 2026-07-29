'''Allowlisted attribute forwarding for env wrappers.

gymnasium 1.0 removed ``Wrapper.__getattr__`` passthrough to the wrapped env,
which this codebase relied on in ~11 places. This restores it for a named set
only, so the forwarded surface stays greppable and a typo raises instead of
silently resolving.
'''


class AttributeForwardingMixin:
    '''Forward the attributes in ``FORWARDED`` to ``self.env``.

    Mix in *before* ``gym.Wrapper`` so this ``__getattr__`` wins.
    '''

    # Attributes call sites read through a wrapper. Extend deliberately.
    FORWARDED = ('GUI', 'CTRL_FREQ', 'PYB_FREQ', 'NAME', 'symbolic', 'state',
                 'constraints', 'done_on_out_of_bound', 'X_GOAL', 'TASK',
                 'denormalize_action', 'normalize_action')

    def __getattr__(self, name):
        if name in type(self).FORWARDED:
            return getattr(self.env, name)
        raise AttributeError(
            f'{type(self).__name__!r} object has no attribute {name!r}; add it '
            f'to FORWARDED if it should pass through to the wrapped env.')
