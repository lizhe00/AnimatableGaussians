class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor, min = None):
        self.initial = initial
        self.interval = interval
        self.factor = factor
        self.min = min

    def get_learning_rate(self, iter_num):
        lr = self.initial * (self.factor ** (iter_num // self.interval))
        if self.min is not None:
            return max(self.min, lr)
        else:
            return lr


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(type, **kwargs):

    if type == 'Step':
        assert 'initial' in kwargs, 'Missing keyword argument "Initial"'
        assert 'interval' in kwargs, 'Missing keyword argument "Interval"'
        assert 'factor' in kwargs, 'Missing keyword argument "Factor"'
        return StepLearningRateSchedule(
                    **kwargs
                )
    elif type == 'Warmup':
        assert 'initial' in kwargs, 'Missing keyword argument "Initial"'
        assert 'final' in kwargs, 'Missing keyword argument "Final"'
        assert 'length' in kwargs, 'Missing keyword argument "Length"'
        return WarmupLearningRateSchedule(
                    kwargs["initial"],
                    kwargs["final"],
                    kwargs["length"],
                )
    elif type == 'Constant':
        assert 'value' in kwargs, 'Missing keyword argument "Value"'
        return ConstantLearningRateSchedule(kwargs["value"])
    else:
        raise ValueError(
            'Unknown learning rate of type "{}"! '
            'Schedule ype must be "Step", "Warmup" or "Constant". '.format(type))
