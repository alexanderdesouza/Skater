"""Feature Importance class"""
from itertools import cycle
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool

from ...data import DataManager
from .base import BaseGlobalInterpretation
from ...util.plotting import COLORS
from ...util.exceptions import *
from ...util.dataops import divide_zerosafe
from ...util.progressbar import ProgressBar
from ...util.static_types import StaticTypes
from ...model.scorer import Scorer


class FeatureImportance(BaseGlobalInterpretation):
    """Contains methods for feature importance. Subclass of BaseGlobalInterpretation.

    """

    def feature_importance(self, model_instance, ascending=True, filter_classes=None, n_jobs=-1,
                           progressbar=True, n_samples=5000, method='output-variance', scorer='default',
                           use_scaling=False):

        """
        Computes feature importance of all features related to a model instance.
        Supports classification, multi-class classification, and regression.

        Wei, Pengfei, Zhenzhou Lu, and Jingwen Song.
        "Variable Importance Analysis: A Comprehensive Review".
        Reliability Engineering & System Safety 142 (2015): 399-432.


        Parameters
        ----------
        model_instance: skater.model.model.Model subtype
            the machine learning model "prediction" function to explain, such that
            predictions = predict_fn(data).
        ascending: boolean, default True
            Helps with ordering Ascending vs Descending
        filter_classes: array type
            The classes to run partial dependence on. Default None invokes all classes.
            Only used in classification models.
        n_jobs: int
            How many concurrent processes to use. Defaults -1, which grabs as many as are available.
            Use 1 to avoid multiprocessing altogether.
        progressbar: bool
            Whether to display progress. This affects which function we use to operate on the pool
            of processes, where including the progress bar results in 10-20% slowdowns.
        n_samples: int
            How many samples to use when computing importance.
        method: string
            How to compute feature importance. conditional-permutation requires Interpretation.training_labels.
            Note this choice should only rarely makes any significant differences

            output-variance: mean absolute value of changes in predictions, given perturbations.

            conditional-permutation: difference in log_loss or MAE of training_labels given perturbations.

        use_scaling: bool
            Whether to weight the importance values by the stregth of the perturbations.

        Returns
        -------
        importances : Sorted Series


        Examples
        --------
            >>> from skater.model import InMemoryModel
            >>> from skater.core.explanations import Interpretation
            >>> from sklearn.ensemble import RandomForestClassier
            >>> rf = RandomForestClassier()
            >>> rf.fit(X,y)
            >>> model = InMemoryModel(rf, examples = X)
            >>> interpreter = Interpretation()
            >>> interpreter.load_data(X)
            >>> interpreter.feature_importance.feature_importance(model)
        """

        if filter_classes:
            err_msg = "members of filter classes must be" \
                      "members of model_instance.classes." \
                      "Expected members of: {0}\n" \
                      "got: {1}".format(model_instance.target_names,
                                        filter_classes)
            filter_classes = list(filter_classes)
            assert all([i in model_instance.target_names for i in filter_classes]), err_msg

        if method == 'conditional-permutation' and self.training_labels is None:
            raise FeatureImportanceError("If interpretation.training_labels are not set, you"
                                         "can only use feature importance methods that do "
                                         "not require ground truth labels")
        elif method == 'conditional-permutation':
            training_labels = self.training_labels.data
        else:
            training_labels = None

        if n_samples <= self.data_set.n_rows:
            inputs = self.data_set.generate_sample(strategy='random-choice',
                                                   sample=True,
                                                   n_samples=n_samples)
        else:
            inputs = self.data_set.data

        original_predictions = model_instance.predict(inputs)
        n_jobs = None if n_jobs < 0 else n_jobs
        predict_fn = model_instance._get_static_predictor()
        executor_instance = Pool(n_jobs)
        arg_list = self.data_set.feature_ids

        if progressbar:
            self.interpreter.logger.warn("Progress bars slow down runs by 10-20%. For slightly \n"
                                         "faster runs, do progress_bar=False")
            n_iter = len(self.data_set.feature_ids)
            p = ProgressBar(n_iter, units='features')
            mapper = executor_instance.imap
        else:
            mapper = executor_instance.map

        fi_func = partial(FeatureImportance.compute_feature_importance,
                          input_data=inputs,
                          estimator_fn=predict_fn,
                          original_predictions=original_predictions,
                          feature_info=self.data_set.feature_info,
                          feature_names=self.data_set.feature_ids,
                          training_labels=training_labels,
                          method=method,
                          scaled=use_scaling,
                          scorer=model_instance.scorers.default)

        importances = {}
        try:
            if n_jobs == 1:
                raise ValueError("Skipping to single processing")
            importance_dicts = []
            for importance in mapper(fi_func, arg_list):
                importance_dicts.append(importance)
                if progressbar:
                    p.animate()
        except:
            self.interpreter.logger.warn("Multiprocessing failed, going single process")
            importance_dicts = []
            for importance in map(fi_func, arg_list):
                importance_dicts.append(importance)
                if progressbar:
                    p.animate()
        finally:
            executor_instance.close()
            executor_instance.join()
            executor_instance.terminate()

        for i in importance_dicts:
            importances.update(i)

        importances = pd.Series(importances).sort_values(ascending=ascending)

        if not importances.sum() > 0:
            self.interpreter.logger.debug("Importances that caused a bug: {}".format(importances))
            raise(FeatureImportanceError("Something went wrong. Importances do not sum to a positive value\n"
                                         "This could be due to:\n"
                                         "1) 0 or infinite divisions\n"
                                         "2) perturbed values == original values\n"
                                         "3) feature is a constant\n"
                                         "Please submit an issue here:\n"
                                         "https://github.com/datascienceinc/Skater/issues"))

        importances = divide_zerosafe(importances, (np.ones(importances.shape[0]) * importances.sum()))
        return importances


    def plot_feature_importance(self, predict_fn, filter_classes=None, ascending=True, ax=None, progressbar=True,
                                n_jobs=-1, n_samples=5000, method='output-variance', use_scaling=False):

        """Computes feature importance of all features related to a model instance,
        then plots the results. Supports classification, multi-class classification, and regression.

        Parameters
        ----------
        predict_fn: skater.model.model.Model subtype
            estimator "prediction" function to explain the predictive model. Could be probability scores
            or target values
        filter_classes: array type
            The classes to run partial dependence on. Default None invokes all classes.
            Only used in classification models.
        ascending: boolean, default True
            Helps with ordering Ascending vs Descending
        ax: matplotlib.axes._subplots.AxesSubplot
            existing subplot on which to plot feature importance. If none is provided,
            one will be created.
        progressbar: bool
            Whether to display progress. This affects which function we use to operate on the pool
            of processes, where including the progress bar results in 10-20% slowdowns.
        n_jobs: int
            How many concurrent processes to use. Defaults -1, which grabs as many as are available.
            Use 1 to avoid multiprocessing altogether.
        n_samples: int
            How many samples to use when computing importance.
        method: string
            How to compute feature importance. conditional-permutation requires Interpretation.training_labels
            output-variance: mean absolute value of changes in predictions, given perturbations.
            conditional-permutation: difference in log_loss or MAE of training_labels given perturbations.
            Note this vary rarely makes any significant differences
        use_scaling: bool
            Whether to weight the importance values by the stregth of the perturbations.


        Returns
        -------
        f: figure instance
        ax: matplotlib.axes._subplots.AxesSubplot
            could be used to for further modification to the plots

        Examples
        --------
            >>> from skater.model import InMemoryModel
            >>> from skater.core.explanations import Interpretation
            >>> from sklearn.ensemble import RandomForestClassier
            >>> rf = RandomForestClassier()
            >>> rf.fit(X,y)
            >>> model = InMemoryModel(rf, examples = X)
            >>> interpreter = Interpretation()
            >>> interpreter.load_data(X)
            >>> interpreter.feature_importance.plot_feature_importance(model, ascending=True, ax=ax)
            """
        try:
            global pyplot
            from matplotlib import pyplot
        except ImportError:
            raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))
        except RuntimeError:
            raise (MatplotlibDisplayError("Matplotlib unable to open display"))

        importances = self.feature_importance(predict_fn,
                                              filter_classes=filter_classes,
                                              progressbar=progressbar,
                                              n_samples=n_samples,
                                              n_jobs=n_jobs,
                                              method=method,
                                              use_scaling=use_scaling)

        if ax is None:
            f, ax = pyplot.subplots(1)
        else:
            f = ax.figure

        colors = cycle(COLORS)
        color = next(colors)
        importances.sort_values(ascending=ascending).plot(kind='barh', ax=ax, color=color)
        return f, ax

    @staticmethod
    def compute_feature_importance(feature_id, input_data, estimator_fn,
                                   original_predictions, feature_info,
                                   feature_names, training_labels=None,
                                   method='output-variance',
                                   scaled=False, scorer=None):
        """Global function for computing column-wise importance

        Parameters
        ----------
        feature_id: hashable
            which feature for which to compute importance
        input_data:
            training data (or sample)
        estimator_fn: callable
            prediction function
        original_predictions: array type
            predicted values before perturbation
        feature_info: dict
            from skater.data.DataManager.feature_info
        feature_names: array type
            list of feature names
        training_labels: array type
            ground truth labels. only required if method="perfomance decrease"
        method: string
            output-variance: importance based on entropy of prediction changes given perturbations
            conditional-permutation: importance based on difference in prediction scores given changes
                                  given perturbations
        model_type: string
            regression or classifition
        scaled: bool
            Whether to weight the importance values by the stregth of the perturbations.

        Returns
        ----------
        importance: dict
            {feature id: importance value}
        """

        copy_of_data_set = DataManager(input_data.copy(), feature_names=feature_names)
        n = copy_of_data_set.n_rows

        original_values = copy_of_data_set[feature_id]

        # collect perturbations
        if feature_info[feature_id]['numeric']:
            samples = copy_of_data_set.generate_column_sample(feature_id, n_samples=n,
                                                              strategy='uniform-over-similarity-ranks')
        else:
            samples = copy_of_data_set.generate_column_sample(feature_id, n_samples=n, strategy='random-choice')


        copy_of_data_set[feature_id] = samples.reshape(-1)

        new_predictions = estimator_fn(copy_of_data_set.values)

        importance = FeatureImportance.compute_importance(new_predictions,
                                                          original_predictions,
                                                          original_values,
                                                          samples,
                                                          training_labels,
                                                          method=method,
                                                          scaled=scaled,
                                                          scorer=scorer)
        return {feature_id: importance}


    @staticmethod
    def compute_importance(new_predictions, original_predictions, original_x, perturbed_x,
                           training_labels, method='output-variance', scaled=False,
                           scorer=None):
        if method == 'output-variance':
            importance = FeatureImportance._compute_importance_via_output_variance(np.array(new_predictions),
                                                                                   np.array(original_predictions),
                                                                                   np.array(original_x),
                                                                                   np.array(perturbed_x),
                                                                                   scaled)
        elif method == 'conditional-permutation':
            importance = FeatureImportance._compute_importance_via_conditional_permutation(np.array(new_predictions),
                                                                                           np.array(original_predictions),
                                                                                           training_labels,
                                                                                           np.array(original_x),
                                                                                           np.array(perturbed_x),
                                                                                           scorer,
                                                                                           scaled)

        else:
            raise(KeyError("Unrecongized method for computing feature_importance: {}".format(method)))
        return importance

    @staticmethod
    def _compute_importance_via_output_variance(new_predictions, original_predictions,
                                                original_x, perturbed_x, scaled=True):
        """Mean absolute change in predictions given perturbations in a feature"""
        changes_in_predictions = abs(new_predictions - original_predictions)
        if len(changes_in_predictions.shape) == 1:
            changes_in_predictions = changes_in_predictions[:, np.newaxis]

        if scaled:
            scales = FeatureImportance.importance_scaler(original_x, perturbed_x)
            changes_in_predictions = np.sum(changes_in_predictions, axis=1) * scales

        importance = np.mean(changes_in_predictions)
        return importance


    @staticmethod
    def _compute_importance_via_conditional_permutation(new_predictions, original_predictions, training_labels,
                                                        original_x, perturbed_x, scorer, scaled=True):

        """Mean absolute error of predictions given perturbations in a feature"""
        if scaled:
            sample_weight = FeatureImportance.importance_scaler(original_x, perturbed_x)
        else:
            sample_weight = None

        score1 = scorer(training_labels, new_predictions, sample_weight=sample_weight)
        score2 = scorer(training_labels, original_predictions, sample_weight=sample_weight)
        multiple = 1 if scorer.type == StaticTypes.scorer_types.increasing else -1
        return abs(max((score2 - score1) * multiple, 0))


    @staticmethod
    def importance_scaler(original_x, perturbed_x):
        perturbed_x = perturbed_x.reshape(original_x.shape)

        scales = abs(perturbed_x - original_x) / (max(original_x) - min(original_x))
        if sum(scales == 0):
            scales = np.ones(perturbed_x.shape[0])

        return scales
