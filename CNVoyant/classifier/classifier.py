import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import progressbar


class Classifier():

    def __init__(self, data_dir: str, normalize_input = True):
        self.random_seed = 42
        self.root_path = os.path.dirname(__file__)
        self.retrained = False
        self.dup_model = None
        self.del_model = None
        self.norm = normalize_input
        self.data_dir = data_dir
        self.model_path = os.path.join(self.data_dir, 'models.pickle')

    def train(self, input: pd.DataFrame, label: str):
        """
        Generate duplication and deletion models based on previously discovered
        optimal hyper-parameters.
        Input: 
            input: pandas DatFrame generate from FeatureBuilder
            label: column title indicating the label of the CNVs
        Output:
            tuple containing a duplication and deletion model
        """

        # Intialize progress bar
        bar_widgets = [progressbar.FormatLabel('Formatting training data'), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.Timer()]
        bar = progressbar.ProgressBar(maxval=3, \
            widgets=bar_widgets)
        bar.start()

        hs_ts_cols = []
        hi_ts = ['HI','TS']
        hi_ts_cols = ['NO_EVIDENCE','LITTLE_EVIDENCE','EMERGING_EVIDENCE',
                'SUFFICIENT_EVIDENCE','AUTOSOMAL_RECESSIVE','UNLIKELY',
                'NOT_EVALUATED']

        for ht in hi_ts:
            for col in hi_ts_cols:
                hs_ts_cols.append(f"{ht}_{col}")

        feature_cols = [
            'BP_LEN','GENE_COUNT','DISEASE_COUNT','EXON_COUNT','REG_COUNT',
            'CENT_DIST','TEL_DIST','%HI','pLI', 'LOEUF','GC_CONTENT','POP_FREQ',
            'CLINVAR_DENSITY','PHYLOP_SCORE','PHASTCONS_SCORE'] + hs_ts_cols

        # Divide training data by CNV type
        XY = input.copy()

        if self.norm:

            log_null_value = 0.1

            XY['BP_LEN'] = np.log(XY['BP_LEN'])
            XY['GENE_COUNT'] = np.log(XY['GENE_COUNT'].replace(0,log_null_value))
            XY['DISEASE_COUNT'] = np.log(XY['DISEASE_COUNT'].replace(0,log_null_value))
            XY['CLINVAR_DENSITY'] = np.log(XY['CLINVAR_DENSITY'].replace(0,log_null_value))

            with open(os.path.join(self.root_path, 'models', 'scalers.pickle'), 'rb') as f:
                self.col_scalers = pickle.load(f)
            for col in feature_cols:
                XY[col] = self.col_scalers[col].transform(XY[col].values.reshape(-1,1))


        XY_dup = XY[XY['CHANGE'] == 'DUP']
        XY_del = XY[XY['CHANGE'] == 'DEL']

        X_dup = XY_dup[feature_cols]
        X_del = XY_del[feature_cols]
        Y_dup = XY_dup[label]
        Y_del = XY_del[label]

        X_dup.columns = [f'f{i}' for i in range(len(X_dup.columns))]
        X_del.columns = [f'f{i}' for i in range(len(X_del.columns))]

        # Map Y to label codes
        def get_map(row, map):
            return map[row]
        
        map = {
            'Benign': 0,
            'VUS': 1,
            'Pathogenic': 2
        }

        Y_del = np.array(Y_del.apply(get_map, map = map))
        Y_dup = np.array(Y_dup.apply(get_map, map = map))

        # Define optimal hyperparameters
        rf_del = {'n_estimators': 275,'min_weight_fraction_leaf': 0.0,
            'min_samples_split': 2,'min_samples_leaf': 3,'max_samples': 1.0,
            'max_features': 'sqrt','max_depth': None,'criterion': 'entropy',
            'class_weight': None,'ccp_alpha': 0.0,'bootstrap': True}

        rf_dup = {'n_estimators': 200,'min_weight_fraction_leaf': 0.0,
            'min_samples_split': 10,'min_samples_leaf': 3,'max_samples': None,
            'max_features': 'sqrt','max_depth': 20,'criterion': 'gini',
            'class_weight': None,'ccp_alpha': 0.0,'bootstrap': False}
        
        
        # Generate models
        bar.update(1)
        bar_widgets[0] = progressbar.FormatLabel('Training deletion model')
        self.del_model = ensemble.RandomForestClassifier(
            **rf_del,
            random_state = self.random_seed
        ).fit(X_del, Y_del)

        self.del_calibrated_model = CalibratedClassifierCV(self.del_model, method='isotonic')
        self.del_calibrated_model.fit(X_del, Y_del)

        bar.update(2)
        bar_widgets[0] = progressbar.FormatLabel('Training duplication model')

        self.dup_model = ensemble.RandomForestClassifier(
            **rf_dup,
            random_state = self.random_seed
        ).fit(X_dup, Y_dup)

        self.dup_calibrated_model = CalibratedClassifierCV(self.dup_model, method='isotonic')
        self.dup_calibrated_model.fit(X_dup, Y_dup)


        bar.update(3)
        bar_widgets[0] = progressbar.FormatLabel('Model training complete')


        self.retrained = True


    def predict(self, input: pd.DataFrame):
        """
        Utilize either the default or re-trained models to return a pandas
        DataFrame containing prediction probabilities for pathogenic, VUS, or
        benign clinical classification.
            Input: pandas DataFrame generated from FeatureBuilder
        """

        # Intialize progress bar
        bar_widgets = [progressbar.FormatLabel('Formatting input data'), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.Timer()]
        bar = progressbar.ProgressBar(maxval=4, \
            widgets=bar_widgets)
        bar.start()

        hs_ts_cols = []
        hi_ts = ['HI','TS']
        hi_ts_cols = ['NO_EVIDENCE','LITTLE_EVIDENCE','EMERGING_EVIDENCE',
                'SUFFICIENT_EVIDENCE','AUTOSOMAL_RECESSIVE','UNLIKELY',
                'NOT_EVALUATED']

        for ht in hi_ts:
            for col in hi_ts_cols:
                hs_ts_cols.append(f"{ht}_{col}")

        feature_cols = [
            'BP_LEN','GENE_COUNT','DISEASE_COUNT','EXON_COUNT','REG_COUNT',
            'CENT_DIST','TEL_DIST','%HI','pLI', 'LOEUF','GC_CONTENT','POP_FREQ',
            'CLINVAR_DENSITY','PHYLOP_SCORE','PHASTCONS_SCORE'] + hs_ts_cols

        # Divide training data by CNV type
        XY = input.copy()

        if self.norm:
            
            log_null_value = 0.1

            XY['BP_LEN'] = np.log(XY['BP_LEN'])
            XY['GENE_COUNT'] = np.log(XY['GENE_COUNT'].replace(0,log_null_value))
            XY['DISEASE_COUNT'] = np.log(XY['DISEASE_COUNT'].replace(0,log_null_value))
            XY['CLINVAR_DENSITY'] = np.log(XY['CLINVAR_DENSITY'].replace(0,log_null_value))

            with open(os.path.join(self.root_path, 'models', 'scalers.pickle'), 'rb') as f:
                self.col_scalers = pickle.load(f)
            for col in feature_cols:
                XY[col] = self.col_scalers[col].transform(XY[col].values.reshape(-1,1))

        XY_dup = XY[XY['CHANGE'] == 'DUP']
        XY_del = XY[XY['CHANGE'] == 'DEL']

        X_dup = XY_dup[feature_cols]
        X_del = XY_del[feature_cols]

        X_dup.columns = [f'f{i}' for i in range(len(X_dup.columns))]
        X_del.columns = [f'f{i}' for i in range(len(X_del.columns))]


        # Generate predictions
        bar_widgets[0] = progressbar.FormatLabel('Obtaining DEL predictions')
        bar.update(1)

        if not self.retrained:
            
            with open(self.model_path, 'rb') as f:
                models = pickle.load(f)

            self.del_model = models['del_model']
            self.del_calibrated_model = models['del_calibrated_model']
            self.dup_model = models['dup_model']
            self.dup_calibrated_model = models['dup_calibrated_model']

        del_pred = self.del_calibrated_model.predict_proba(X_del)
        del_pred = pd.DataFrame(del_pred).rename(columns = {0:'BENIGN',1:'VUS',2:'PATHOGENIC'})

        bar_widgets[0] = progressbar.FormatLabel('Obtaining DUP predictions')
        bar.update(2)

        dup_pred = self.dup_calibrated_model.predict_proba(X_dup)
        dup_pred = pd.DataFrame(dup_pred).rename(columns = {0:'BENIGN',1:'VUS',2:'PATHOGENIC'})

        bar_widgets[0] = progressbar.FormatLabel('Reformatting and saving predictions')
        bar.update(3)
        
        # Concatenate positions and predictions
        XY_dup_orig = XY_dup[[ c for c in XY_dup.columns if c not in feature_cols + list(dup_pred.columns) ]]
        XY_del_orig = XY_del[[ c for c in XY_del.columns if c not in feature_cols + list(dup_pred.columns) ]]

        XY_dup_orig = XY_dup_orig.reset_index(drop=True)
        X_dup = X_dup.reset_index(drop=True)
        X_dup.columns = feature_cols
        dup_pred = dup_pred.reset_index(drop=True)
        dup_res = pd.concat([XY_dup_orig,X_dup,dup_pred], axis = 1)

        XY_del_orig = XY_del_orig.reset_index(drop=True)
        X_del = X_del.reset_index(drop=True)
        X_del.columns = feature_cols
        del_pred = del_pred.reset_index(drop=True)
        del_res = pd.concat([XY_del_orig,X_del,del_pred], axis = 1)

        # Create a prediction column
        def get_pred(row):
            pred_dict = {x:row[x] for x in ['BENIGN','VUS','PATHOGENIC']}
            pred_max = max(pred_dict.values())
            for k,v in pred_dict.items():
                if v == pred_max:
                    return k
        
        if len(del_res.index) != 0:
            del_res['PREDICTION'] = del_res.apply(get_pred, axis = 1)

        if len(dup_res.index) != 0:
            dup_res['PREDICTION'] = dup_res.apply(get_pred, axis = 1)
        
        bar_widgets[0] = progressbar.FormatLabel('Predictions complete')
        bar.update(4)


        return pd.concat([del_res, dup_res]).sort_values(['CHROMOSOME','START']).rename(columns={'gene_info':'INTERSECTING_GENES'})
    
    
    
        