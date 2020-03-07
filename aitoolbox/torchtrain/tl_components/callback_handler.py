import torch

from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.utils.util import is_empty_function


class BasicCallbacksHandler:
    def __init__(self, train_loop_obj):
        """Callback handler used for the callback orchestration inside the TrainLoop

        Common use of this handler is to call different methods inside the TrainLoop at different stages of the training
        process. Thus execute desired callbacks' functionality at the desired point of the training process.

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop
        """
        self.train_loop_obj = train_loop_obj

    def register_callbacks(self, callbacks):
        """Register TrainLoop object reference inside the listed callbacks when the TrainLoop is created

        Normally, this is called from inside of the train loop by the TrainLoop itself. Basically train loop "registers"
        itself.

        Args:
            callbacks (list or None): list of callbacks

        Returns:
            None
        """
        if callbacks is not None and len(callbacks) > 0:
            self.enforce_callbacks_quality(callbacks)
            self.train_loop_obj.callbacks += [cb.register_train_loop_object(self.train_loop_obj) for cb in callbacks]

        if not all(0 == cb.execution_order for cb in self.train_loop_obj.callbacks):
            self.train_loop_obj.callbacks = sorted(self.train_loop_obj.callbacks, key=lambda cb: cb.execution_order)

    def execute_epoch_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_epoch_begin()

    def execute_epoch_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_epoch_end()

    def execute_train_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_train_begin()

    def execute_train_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_train_end()

    def execute_batch_begin(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_batch_begin()

    def execute_batch_end(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_batch_end()

    def execute_gradient_update(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_after_gradient_update()

    def execute_optimizer_step(self):
        for callback in self.train_loop_obj.callbacks:
            callback.on_after_optimizer_step()

    def enforce_callbacks_quality(self, callbacks):
        for cb in callbacks:
            if not isinstance(cb, AbstractCallback):
                raise TypeError(f'Callback {cb} is not inherited from the AbstractCallback')
            
            if cb.device_idx_execution is not None and self.train_loop_obj.device.index is not None:
                if cb.device_idx_execution >= torch.cuda.device_count():
                    raise ValueError(f'Selected device_idx_execution of {cb.device_idx_execution} is too high. '
                                     f'There are only {torch.cuda.device_count()} available GPU devices. '
                                     f'Select index ranging from 0 to {torch.cuda.device_count() - 1}')

    @staticmethod
    def print_callback_info(callback_list):
        return '\n'.join([f'\t{callback.callback_name}: {type(callback)}, execution_order: {callback.execution_order}'
                          for callback in callback_list])

    def print_registered_callback_names(self):
        print(self)

    def __str__(self):
        return 'CALLBACKS:\n' + self.print_callback_info(self.train_loop_obj.callbacks)

    def __len__(self):
        return len(self.train_loop_obj.callbacks)

    def __add__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            BasicCallbacksHandler:
        """
        self.register_callbacks(other)
        return self

    def __iadd__(self, other):
        """

        Args:
            other (list): callbacks list

        Returns:
            BasicCallbacksHandler:
        """
        self.register_callbacks(other)
        return self

    def __contains__(self, item):
        """

        Args:
            item:

        Returns:
            bool:
        """
        if type(item) == str:
            for cb in self.train_loop_obj.callbacks:
                if cb.callback_name == item:
                    return True
        else:
            for cb in self.train_loop_obj.callbacks:
                if type(cb) == item:
                    return True
        return False


class CallbacksHandler(BasicCallbacksHandler):
    def __init__(self, train_loop_obj):
        """Callback handler used for the callback orchestration inside the TrainLoop

        Compared to `BasicCallbacksHandler`, this handler will at certain TrainLoop stage only execute those
        callbacks which have implemented the functionality intended to be executed at this particular stage.
        Thus, `CallbacksHandler` doesn't unnecessarily execute callbacks at stages they are not implemented at.

        Common use of this handler is to call different methods inside the TrainLoop at different stages of the training
        process. Thus execute desired callbacks' functionality at the desired point of the training process.

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop
        """
        super().__init__(train_loop_obj)

        self.cbs_on_epoch_begin = []
        self.cbs_on_epoch_end = []
        self.cbs_on_train_begin = []
        self.cbs_on_train_end = []
        self.cbs_on_batch_begin = []
        self.cbs_on_batch_end = []
        self.cbs_on_after_gradient_update = []
        self.cbs_on_after_optimizer_step = []
        
        self.registered_cbs = [
            self.cbs_on_epoch_begin, self.cbs_on_epoch_end,
            self.cbs_on_train_begin, self.cbs_on_train_end,
            self.cbs_on_batch_begin,  self.cbs_on_batch_end,
            self.cbs_on_after_gradient_update, self.cbs_on_after_optimizer_step
        ]

    def register_callbacks(self, callbacks):
        """Register TrainLoop object reference inside the listed callbacks when the TrainLoop is created

        Normally, this is called from inside of the train loop by the TrainLoop itself. Basically train loop "registers"
        itself.

        Args:
            callbacks (list or None): list of callbacks

        Returns:
            None
        """
        super().register_callbacks(callbacks)
        self.split_on_execution_position(callbacks, register_train_loop=False)

    def execute_epoch_begin(self):
        for callback in self.cbs_on_epoch_begin:
            callback.on_epoch_begin()

    def execute_epoch_end(self):
        for callback in self.cbs_on_epoch_end:
            callback.on_epoch_end()

    def execute_train_begin(self):
        for callback in self.cbs_on_train_begin:
            callback.on_train_begin()

    def execute_train_end(self):
        for callback in self.cbs_on_train_end:
            callback.on_train_end()

    def execute_batch_begin(self):
        for callback in self.cbs_on_batch_begin:
            callback.on_batch_begin()

    def execute_batch_end(self):
        for callback in self.cbs_on_batch_end:
            callback.on_batch_end()

    def execute_gradient_update(self):
        for callback in self.cbs_on_after_gradient_update:
            callback.on_after_gradient_update()

    def execute_optimizer_step(self):
        for callback in self.cbs_on_after_optimizer_step:
            callback.on_after_optimizer_step()

    def split_on_execution_position(self, callbacks, register_train_loop=False):
        if callbacks is not None and len(callbacks) > 0:
            for callback in callbacks:
                if register_train_loop:
                    callback = callback.register_train_loop_object(self.train_loop_obj)

                if not is_empty_function(callback.on_epoch_begin):
                    self.cbs_on_epoch_begin.append(callback)

                if not is_empty_function(callback.on_epoch_end):
                    self.cbs_on_epoch_end.append(callback)

                if not is_empty_function(callback.on_train_begin):
                    self.cbs_on_train_begin.append(callback)

                if not is_empty_function(callback.on_train_end):
                    self.cbs_on_train_end.append(callback)

                if not is_empty_function(callback.on_batch_begin):
                    self.cbs_on_batch_begin.append(callback)

                if not is_empty_function(callback.on_batch_end):
                    self.cbs_on_batch_end.append(callback)

                if not is_empty_function(callback.on_after_gradient_update):
                    self.cbs_on_after_gradient_update.append(callback)

                if not is_empty_function(callback.on_after_optimizer_step):
                    self.cbs_on_after_optimizer_step.append(callback)

        for cbs_at_position in self.registered_cbs:
            if not all(0 == cb.execution_order for cb in cbs_at_position):
                cbs_at_position.sort(key=lambda cb: cb.execution_order)

    def __str__(self):
        return 'CALLBACKS\n' \
               f'At on_epoch_begin:\n{self.print_callback_info(self.cbs_on_epoch_begin)}\n' \
               f'At on_epoch_end:\n{self.print_callback_info(self.cbs_on_epoch_end)}\n' \
               f'At on_train_begin:\n{self.print_callback_info(self.cbs_on_train_begin)}\n' \
               f'At on_train_end:\n{self.print_callback_info(self.cbs_on_train_end)}\n' \
               f'At on_batch_begin:\n{self.print_callback_info(self.cbs_on_batch_begin)}\n' \
               f'At on_batch_end:\n{self.print_callback_info(self.cbs_on_batch_end)}\n' \
               f'At on_after_gradient_update:\n{self.print_callback_info(self.cbs_on_after_gradient_update)}\n' \
               f'At on_after_optimizer_step:\n{self.print_callback_info(self.cbs_on_after_optimizer_step)}'
