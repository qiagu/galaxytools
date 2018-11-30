import imblearn
import json
import numpy as np
import os
import pandas
import pickle
import skrebate
import sklearn
import sys
import xgboost
import warnings
from imblearn import under_sampling, over_sampling, combine
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn import (cluster, decomposition, ensemble, feature_extraction, feature_selection,
                    gaussian_process, kernel_approximation, metrics,
                    model_selection, naive_bayes, neighbors, pipeline, preprocessing,
                    svm, linear_model, tree, discriminant_analysis)
from sklearn.exceptions import FitFailedWarning
from sklearn.externals import joblib
from utils import get_cv, get_scoring, get_X_y, load_model, read_columns, SafeEval


N_JOBS = int(os.environ.get('GALAXY_SLOTS', 1))


def get_search_params(params_builder):
    search_params = {}
    safe_eval = SafeEval(load_scipy=True, load_numpy=True)
    safe_eval_es = SafeEval(load_estimators=True)

    for p in params_builder['param_set']:
        search_p = p['search_param_selector']['search_p']
        if search_p.strip() == '':
            continue
        param_type = p['search_param_selector']['selected_param_type']

        lst = search_p.split(":")
        assert (len(lst) == 2), "Error, make sure there is one and only one colon in search parameter input."
        literal = lst[1].strip()
        param_name = lst[0].strip()
        if param_name:
            if param_name.lower() == 'n_jobs':
                sys.exit("Parameter `%s` is invalid for search." %param_name)
            elif not param_name.endswith('-'):
                ev = safe_eval(literal)
                if param_type == "final_estimator_p":
                    search_params["estimator__" + param_name] = ev
                else:
                    search_params["preprocessing_" + param_type[5:6] + "__" + param_name] = ev
            else:
                # only for estimator eval, add `-` to the end of param
                #TODO maybe add regular express check
                ev = safe_eval_es(literal)
                for obj in ev:
                    if 'n_jobs' in obj.get_params():
                        obj.set_params( n_jobs=N_JOBS )
                if param_type == "final_estimator_p":
                    search_params["estimator__" + param_name[:-1]] = ev
                else:
                    search_params["preprocessing_" + param_type[5:6] + "__" + param_name[:-1]] = ev
        elif param_type != "final_estimator_p":
            #TODO regular express check ?
            ev = safe_eval_es(literal)
            preprocessors = [preprocessing.StandardScaler(), preprocessing.Binarizer(), preprocessing.Imputer(),
                            preprocessing.MaxAbsScaler(), preprocessing.Normalizer(), preprocessing.MinMaxScaler(),
                            preprocessing.PolynomialFeatures(),preprocessing.RobustScaler(),
                            feature_selection.SelectKBest(), feature_selection.GenericUnivariateSelect(),
                            feature_selection.SelectPercentile(), feature_selection.SelectFpr(), feature_selection.SelectFdr(),
                            feature_selection.SelectFwe(), feature_selection.VarianceThreshold(),
                            decomposition.FactorAnalysis(random_state=0), decomposition.FastICA(random_state=0), decomposition.IncrementalPCA(),
                            decomposition.KernelPCA(random_state=0, n_jobs=N_JOBS), decomposition.LatentDirichletAllocation(random_state=0, n_jobs=N_JOBS),
                            decomposition.MiniBatchDictionaryLearning(random_state=0, n_jobs=N_JOBS),
                            decomposition.MiniBatchSparsePCA(random_state=0, n_jobs=N_JOBS), decomposition.NMF(random_state=0),
                            decomposition.PCA(random_state=0), decomposition.SparsePCA(random_state=0, n_jobs=N_JOBS),
                            decomposition.TruncatedSVD(random_state=0),
                            kernel_approximation.Nystroem(random_state=0), kernel_approximation.RBFSampler(random_state=0),
                            kernel_approximation.AdditiveChi2Sampler(), kernel_approximation.SkewedChi2Sampler(random_state=0),
                            cluster.FeatureAgglomeration(),
                            skrebate.ReliefF(n_jobs=N_JOBS), skrebate.SURF(n_jobs=N_JOBS), skrebate.SURFstar(n_jobs=N_JOBS),
                            skrebate.MultiSURF(n_jobs=N_JOBS), skrebate.MultiSURFstar(n_jobs=N_JOBS),
                            imblearn.under_sampling.ClusterCentroids(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.CondensedNearestNeighbour(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.EditedNearestNeighbours(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.RepeatedEditedNearestNeighbours(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.AllKNN(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.InstanceHardnessThreshold(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.NearMiss(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.NeighbourhoodCleaningRule(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.OneSidedSelection(random_state=0, n_jobs=N_JOBS),
                            imblearn.under_sampling.RandomUnderSampler(random_state=0),
                            imblearn.under_sampling.TomekLinks(random_state=0, n_jobs=N_JOBS),
                            imblearn.over_sampling.ADASYN(random_state=0, n_jobs=N_JOBS),
                            imblearn.over_sampling.RandomOverSampler(random_state=0),
                            imblearn.over_sampling.SMOTE(random_state=0, n_jobs=N_JOBS),
                            imblearn.over_sampling.SVMSMOTE(random_state=0, n_jobs=N_JOBS),
                            imblearn.over_sampling.BorderlineSMOTE(random_state=0, n_jobs=N_JOBS),
                            imblearn.over_sampling.SMOTENC(categorical_features=[], random_state=0, n_jobs=N_JOBS),
                            imblearn.combine.SMOTEENN(random_state=0), imblearn.combine.SMOTETomek(random_state=0)]
            newlist = []
            for obj in ev:
                if obj is None:
                    newlist.append(None)
                elif obj == 'all_0':
                    newlist.extend(preprocessors[0:36])
                elif obj == 'sk_prep_all':      # no KernalCenter()
                    newlist.extend(preprocessors[0:8])
                elif obj == 'fs_all':
                    newlist.extend(preprocessors[8:15])
                elif obj == 'decomp_all':
                    newlist.extend(preprocessors[15:26])
                elif obj == 'k_appr_all':
                    newlist.extend(preprocessors[26:30])
                elif obj == "reb_all":
                    newlist.extend(preprocessors[31:36])
                elif obj == 'imb_all':
                    newlist.extend(preprocessors[36:55])
                elif  type(obj) is int and -1 < obj < len(preprocessors):
                    newlist.append(preprocessors[obj])
                elif hasattr(obj, 'get_params'):       # user object
                    if 'n_jobs' in obj.get_params():
                        newlist.append( obj.set_params(n_jobs=N_JOBS) )
                    else:
                        newlist.append(obj)
                else:
                    sys.exit("Unsupported preprocessor type: %r" %(obj))
            search_params["preprocessing_" + param_type[5:6]] = newlist
        else:
            sys.exit("Parameter name of the final estimator can't be skipped!")

    return search_params


if __name__ == '__main__':

    warnings.simplefilter('ignore')

    input_json_path = sys.argv[1]
    with open(input_json_path, "r") as param_handler:
        params = json.load(param_handler)

    infile_pipeline = sys.argv[2]
    infile1 = sys.argv[3]
    infile2 = sys.argv[4]
    outfile_result = sys.argv[5]
    if len(sys.argv) > 6:
        outfile_estimator = sys.argv[6]
    else:
        outfile_estimator = None

    params_builder = params['search_schemes']['search_params_builder']

    input_type = params["input_options"]["selected_input"]
    if input_type=="tabular":
        header = 'infer' if params["input_options"]["header1"] else None
        column_option = params["input_options"]["column_selector_options_1"]["selected_column_selector_option"]
        if column_option in ["by_index_number", "all_but_by_index_number", "by_header_name", "all_but_by_header_name"]:
            c = params["input_options"]["column_selector_options_1"]["col1"]
        else:
            c = None
        X = read_columns(
                infile1,
                c = c,
                c_option = column_option,
                sep='\t',
                header=header,
                parse_dates=True
        )
    else:
        X = mmread(open(infile1, 'r'))

    header = 'infer' if params["input_options"]["header2"] else None
    column_option = params["input_options"]["column_selector_options_2"]["selected_column_selector_option2"]
    if column_option in ["by_index_number", "all_but_by_index_number", "by_header_name", "all_but_by_header_name"]:
        c = params["input_options"]["column_selector_options_2"]["col2"]
    else:
        c = None
    y = read_columns(
            infile2,
            c = c,
            c_option = column_option,
            sep='\t',
            header=header,
            parse_dates=True
    )
    y=y.ravel()

    optimizers = params["search_schemes"]["selected_search_scheme"]
    optimizers = getattr(model_selection, optimizers)

    options = params["search_schemes"]["options"]
    splitter, groups = get_cv(options.pop('cv_selector'))
    if groups is None:
        options['cv'] = splitter
    elif groups == "":
        options['cv'] = list( splitter.split(X, y, groups=None) )
    else:
        options['cv'] = list( splitter.split(X, y, groups=groups) )
    options['n_jobs'] = N_JOBS
    primary_scoring = options['scoring']['primary_scoring']
    options['scoring'] = get_scoring(options['scoring'])
    if options['error_score']:
        options['error_score'] = 'raise'
    else:
        options['error_score'] = np.NaN
    if options['refit'] and isinstance(options['scoring'], dict):
        options['refit'] = 'primary'
    if 'pre_dispatch' in options and options['pre_dispatch'] == '':
        options['pre_dispatch'] = None

    with open(infile_pipeline, 'rb') as pipeline_handler:
        pipeline = load_model(pipeline_handler)

    cache_scheme = params['cache_selector']
    cache_dir = './memory_cache'
    ## TODO add user cache
    memory = joblib.Memory(cache_dir, verbose=0)


    class MemoryFit(object):
        def fit(self, *args, **kwargs):
            fit = memory.cache(super(MemoryFit, self).fit)
            cached_self = fit(*args, **kwargs)
            vars(self).update(vars(cached_self))


    ## TODO make dynamic cache classes
    class CachedAdaBoostClassifier(MemoryFit, ensemble.AdaBoostClassifier): pass
    class CachedAdaBoostRegressor(MemoryFit, ensemble.AdaBoostRegressor): pass
    class CachedExtraTreesClassifier(MemoryFit, ensemble.ExtraTreesClassifier): pass
    class CachedExtraTreesRegressor(MemoryFit, ensemble.ExtraTreesRegressor): pass
    class CachedGradientBoostingClassifier(MemoryFit, ensemble.GradientBoostingClassifier): pass
    class CachedGradientBoostingRegressor(MemoryFit, ensemble.GradientBoostingRegressor): pass 
    class CachedRandomForestClassifier(MemoryFit, ensemble.RandomForestClassifier): pass
    class CachedRandomForestRegressor(MemoryFit, ensemble.RandomForestRegressor): pass

    class CachedLasso(MemoryFit, linear_model.Lasso): pass
    class CachedLinearRegression(MemoryFit, linear_model.LinearRegression): pass
    class CachedLogisticRegression(MemoryFit, linear_model.LogisticRegression): pass
    class CachedPassiveAggressiveClassifier(MemoryFit, linear_model.PassiveAggressiveClassifier): pass
    class CachedPassiveAggressiveRegressor(MemoryFit, linear_model.PassiveAggressiveRegressor): pass
    class CachedPerceptron(MemoryFit, linear_model.Perceptron): pass
    class CachedRidgeClassifier(MemoryFit, linear_model.RidgeClassifier): pass
    class CachedSGDClassifier(MemoryFit, linear_model.SGDClassifier): pass
    class CachedSGDRegressor(MemoryFit, linear_model.SGDRegressor): pass

    
    class CachedBernoulliNB(MemoryFit, naive_bayes.BernoulliNB): pass
    class CachedGaussianNB(MemoryFit, naive_bayes.GaussianNB): pass
    class CachedMultinomialNB(MemoryFit, naive_bayes.MultinomialNB): pass

    class CachedKNeighborsClassifier(MemoryFit, neighbors.KNeighborsClassifier): pass
    class CachedKNeighborsRegressor(MemoryFit, neighbors.KNeighborsRegressor): pass

    class CachedLinearSVC(MemoryFit, svm.LinearSVC): pass
    class CachedLinearSVR(MemoryFit, svm.LinearSVR): pass
    class CachedNuSVC(MemoryFit, svm.NuSVC): pass
    class CachedNuSVR(MemoryFit, svm.NuSVR): pass
    class CachedSVC(MemoryFit, svm.SVC): pass
    class CachedSVR(MemoryFit, svm.SVR): pass

    class CachedDecisionTreeClassifier(MemoryFit, tree.DecisionTreeClassifier): pass
    class CachedDecisionTreeRegressor(MemoryFit, tree.DecisionTreeRegressor): pass
    class CachedExtraTreeClassifier(MemoryFit, tree.ExtraTreeClassifier): pass
    class CachedExtraTreeRegressor(MemoryFit, tree.ExtraTreeRegressor): pass
    
    class CachedXGBClassifier(MemoryFit, xgboost.XGBClassifier): pass
    class CachedXGBRegressor(MemoryFit, xgboost.XGBRegressor): pass

    cache_mapper = {
        'CachedAdaBoostClassifier': CachedAdaBoostClassifier,
        'CachedAdaBoostRegressor': CachedAdaBoostRegressor,
        'CachedExtraTreesClassifier': CachedExtraTreesClassifier,
        'CachedExtraTreesRegressor': CachedExtraTreesRegressor,
        'CachedGradientBoostingClassifier': CachedGradientBoostingClassifier,
        'CachedGradientBoostingRegressor': CachedGradientBoostingRegressor,
        'CachedRandomForestClassifier': CachedRandomForestClassifier,
        'CachedRandomForestRegressor': CachedRandomForestRegressor,
        'CachedLasso': CachedLasso,
        'CachedLinearRegression': CachedLinearRegression,
        'CachedLogisticRegression': CachedLogisticRegression,
        'CachedPassiveAggressiveClassifier': CachedPassiveAggressiveClassifier,
        'CachedPassiveAggressiveRegressor': CachedPassiveAggressiveRegressor,
        'CachedPerceptron': CachedPerceptron,
        'CachedRidgeClassifier': CachedRidgeClassifier,
        'CachedSGDClassifier': CachedSGDClassifier,
        'CachedSGDRegressor': CachedSGDRegressor,
        'CachedBernoulliNB': CachedBernoulliNB,
        'CachedGaussianNB': CachedGaussianNB,
        'CachedMultinomialNB': CachedMultinomialNB,
        'CachedKNeighborsClassifier': CachedKNeighborsClassifier,
        'CachedKNeighborsRegressor': CachedKNeighborsRegressor,
        'CachedLinearSVC': CachedLinearSVC,
        'CachedLinearSVR': CachedLinearSVR,
        'CachedNuSVC': CachedNuSVC,
        'CachedNuSVR': CachedNuSVR,
        'CachedSVC': CachedSVC,
        'CachedSVR': CachedSVR,
        'CachedDecisionTreeClassifier': CachedDecisionTreeClassifier,
        'CachedDecisionTreeRegressor': CachedDecisionTreeRegressor,
        'CachedExtraTreeClassifier': CachedExtraTreeClassifier,
        'CachedExtraTreeRegressor': CachedExtraTreeRegressor,
        'CachedXGBClassifier': CachedXGBClassifier,
        'CachedXGBRegressor': CachedXGBRegressor
    }


    def cache_object(obj):
        """
        Turn an object into cached
        """
        params = obj.get_params()
        if 'estimator' not in params: 
            klass = obj.__class__
            cache_class = 'Cached' + klass.__name__
            new_class = type(cache_class, (MemoryFit, klass), {})
            obj.__class__ = new_class
            return obj
        # else cache the estimator instead
        estimator = params['estimator']
        cached_estimator = cache_object(estimator)
        obj.set_params( estimator=cached_estimator )
        return obj


    def cache_estimator(est):
        """
        Turn an estimator into cached
        """
        klass_name = est.__class__.__name__
        est.__class__ = cache_mapper['Cached' + klass_name]
        return est


    def get_cached_pipeline(input_json, pipeline):
        """
        input_json: e.g.  {"selected_cache_scheme": "within_job", "cache_steps": "-1, 0"}
        pipeline: sklearn Pipeline object

        return: Pipeline object with cache
        """
        pipeline_steps = pipeline.get_params()['steps']
        cache_steps = input_json['cache_steps'].split(',')
        cache_steps = [int(x.strip()) for x in cache_steps]
        for step in cache_steps:
            if step == -1: # final estimator
                estimator = pipeline_steps[-1][-1]
                pipeline_steps[-1] = ('estimator', cache_object(estimator))
            else: # pre-processing transformers
                try:
                    transformer = pipeline_steps[:-1][step][-1]
                    name = pipeline_steps[:-1][step][0]
                    pipeline_steps[step] = (name, cache_object(transformer))
                except IndexError:
                    sys.exit("Pre-processing step #%d doesn't exist!" %step)
        pipeline.set_params(steps=pipeline_steps)
        return pipeline


    def get_cached_pipeline_simplified(input_json, pipeline):
        """
        input_json: e.g.  {"selected_cache_scheme": "within_job"}
        pipeline: sklearn Pipeline object

        return: Pipeline object with cache
        """
        pipeline_steps = pipeline.get_params()['steps']
        for i in range(len(pipeline_steps)):
            name, transformer = pipeline_steps[i]
            params = transformer.get_params()
            if 'estimator' in params:
                cached_estimator = cache_estimator(params['estimator'])
                transformer.set_params(estimator=cached_estimator)
            pipeline_steps[i] = (name, transformer)
        pipeline.set_params(memory=memory, steps=pipeline_steps)
        return pipeline


    if cache_scheme['selected_cache_scheme'] != 'nope':
        pipeline = get_cached_pipeline_simplified(cache_scheme, pipeline)

    search_params = get_search_params(params_builder)
    searcher = optimizers(pipeline, search_params, **options)

    if options['error_score'] == 'raise':
        searcher.fit(X, y)
    else:
        warnings.simplefilter('always', FitFailedWarning)
        with warnings.catch_warnings(record=True) as w:
            try:
                searcher.fit(X, y)
            except ValueError:
                pass
            for warning in w:
                print(repr(warning.message))

    cv_result = pandas.DataFrame(searcher.cv_results_)
    cv_result.rename(inplace=True, columns={"mean_test_primary": "mean_test_"+primary_scoring, "rank_test_primary": "rank_test_"+primary_scoring})
    cv_result.to_csv(path_or_buf=outfile_result, sep='\t', header=True, index=False)

    if outfile_estimator:
        with open(outfile_estimator, "wb") as output_handler:
            pickle.dump(searcher.best_estimator_, output_handler, pickle.HIGHEST_PROTOCOL)
