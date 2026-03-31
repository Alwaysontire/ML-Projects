import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * pow((self.s0/(self.s0 + iteration)), self.p)


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        ...
        self.model.loss_history.append(self.model.compute_loss())
        while self.iteration < self.max_iter:
            delta = self._step()
            self.model.loss_history.append(self.model.compute_loss())
            if delta.T @ delta < self.tolerance or np.isnan(delta).sum() > 0:
                break
        # в конце также приcваивает атрибуту модели полученный loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train
        w_k = self.model.w

        gradient = self.model.compute_gradients(X_train, y_train)
        self.model.w = self.model.w - self.lr_schedule.get_lr(self.iteration) * gradient
        delta = self.model.w - w_k
        
        return delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        batch_index = np.random.randint(0, self.model.X_train.shape[0], size=self.batch_size)
        X_batch = self.model.X_train[batch_index]
        y_batch = self.model.y_train[batch_index]
        w_k = self.model.w

        gradients = self.model.compute_gradients(X_batch, y_batch)
        self.model.w = self.model.w - self.lr_schedule.get_lr(self.iteration) * gradients
        
        return self.model.w - w_k


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape
        
        if self.grad_memory is None:
            ...
            # TODO: инициализировать хранилища при первом вызове
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)

        # TODO: реализовать SAG
        batch_index = np.random.randint(0, num_objects, size=self.batch_size)
        for idx in batch_index:
            X_batch = X_train[idx: idx + 1]
            y_batch = y_train[idx: idx + 1]
            g_new = self.model.compute_gradients(X_batch, y_batch)
            g_old = self.grad_memory[idx]
            self.grad_sum += (g_new - g_old)/num_objects
            self.grad_memory[idx] = g_new
            

        w_k = self.model.w
        self.model.w = w_k - self.lr_schedule.get_lr(self.iteration) * self.grad_sum

        return self.model.w - w_k


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать градиентный спуск с моментумом
        if self.velocity is None:
            self.velocity = np.zeros(self.model.w.shape[0])
        gradients = self.model.compute_gradients()
        self.velocity = self.beta * self.velocity + self.lr_schedule.get_lr(self.iteration) * gradients

        w_k = self.model.w
        self.model.w = w_k - self.velocity
        return self.model.w - w_k
        

class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать Adam по формуле из ноутбука

        if self.m is None and self.v is None:
            self.m = np.zeros(self.model.w.shape[0])
            self.v = np.zeros(self.model.w.shape[0])

        gradients = self.model.compute_gradients()

        m_k = self.m
        self.m = self.beta1 * m_k + (1 - self.beta1) * gradients
        v_k = self.v
        self.v = self.beta2 * v_k + (1 - self.beta2) * np.power(gradients, 2)

        m_estim = self.m / (1 - np.power(self.beta1, self.iteration + 1))
        v_estim = self.v / (1 - np.power(self.beta2, self.iteration + 1))

        w_k = self.model.w

        self.model.w = w_k - self.lr_schedule.get_lr(self.iteration) * m_estim / (np.sqrt(v_estim) + self.eps)
        
        return self.model.w - w_k




# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        X = self.model.X_train
        y = self.model.y_train
        w_opt = self.model.loss_function.analytic_solution_func(X, y)
        self.model.w = w_opt



