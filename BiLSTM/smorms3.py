from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K


class SMORMS3(Optimizer):
    """SMORMS3 optimizer.

    Default parameters follow those provided in the blog post.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [RMSprop loses to SMORMS3 - Beware the Epsilon!](http://sifter.org/~simon/journal/20150420.html)
    """

    def __init__(self, lr=0.001, epsilon=1e-16, decay=0.,
                 **kwargs):
        super(SMORMS3, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        mems = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs + mems
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))


        for p, g, m, v, mem in zip(params, grads, ms, vs, mems):

            r = 1. / (1. + mem)
            new_m = (1. - r) * m + r * g
            new_v = (1. - r) * v + r * K.square(g)
            denoise = K.square(new_m) / (new_v + self.epsilon)
            new_p = p - g * K.minimum(lr, denoise) / (K.sqrt(new_v) + self.epsilon)
            new_mem = 1. + mem * (1. - denoise)

            self.updates.append(K.update(m, new_m))
            self.updates.append(K.update(v, new_v))
            self.updates.append(K.update(mem, new_mem))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(SMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
