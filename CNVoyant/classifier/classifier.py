import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import ensemble
import xgboost
from sklearn.pipeline import Pipeline
import onnxruntime as rt
import progressbar


class Classifier():

    def __init__(self, normalize_input = True):
        self.random_seed = 42
        self.root_path = os.path.dirname(__file__)
        self.onnx = True
        self.dup_model = None
        self.del_model = None
        self.norm = normalize_input

    def train(self, input: pd.DataFrame):
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

        # Limit to feature, label, and CNV type columns
        # feature_cols = [
        #     'BP_LEN','GENE_COUNT','DISEASE_COUNT','CENT_DIST','TEL_DIST', 
        #     'HI_SCORE','TS_SCORE','%HI','pLI','LOEUF','GC_CONTENT', 
        #     'POP_FREQ','CLINVAR_DENSITY','P_TRIPLO','P_HAPLO','gnomad_oe_lof',
        #     'gnomad_oe_lof_upper','gnomad_pLI','gnomad_lof_z','gnomad_mis_z',
        #     'gnomad_oe_mis_upper','episcore','sHet','exon_phastcons','hurles_hi',
        #     'rvis','promoter_cpg_count','eds','promoter_phastcons','cds_length',
        #     'gnomad_pNull'
        # ]
        feature_cols = [
            'BP_LEN','GENE_COUNT','DISEASE_COUNT','CENT_DIST','TEL_DIST', 
            'HI_SCORE','TS_SCORE','%HI','pLI','LOEUF','GC_CONTENT', 
            'POP_FREQ','CLINVAR_DENSITY'
        ]
        label = 'LABEL'

        # Divide training data by CNV type
        XY = input.copy()

        if self.norm:
            with open(os.path.join(self.root_path, 'models', 'scalers.pickle'), 'rb') as f:
                col_scalers = pickle.load(f)
            for col in feature_cols:
                XY[col] = col_scalers[col].transform(XY[col].values.reshape(-1,1))

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
        rf_del = {
            'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 3, 
            'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False
            }

        xgb_dup = {
            'tree_method': 'approx', 'reg_lambda': 1, 'reg_alpha': 0,
            'n_estimators': 100, 'max_depth': 10, 'gamma': 0, 'eta': 0.1, 
            'booster': 'gbtree', 'base_score': 0.5
         }
        
        # Generate models
        bar.update(1)
        bar_widgets[0] = progressbar.FormatLabel('Training deletion model')
        self.del_model = ensemble.RandomForestClassifier(
            **rf_del,
            random_state = self.random_seed
        ).fit(X_del, Y_del)
        bar.update(2)
        bar_widgets[0] = progressbar.FormatLabel('Training duplication model')

        self.dup_model = xgboost.XGBClassifier(
            **xgb_dup,
            random_state = self.random_seed
        ).fit(X_dup, Y_dup)
        bar.update(3)
        bar_widgets[0] = progressbar.FormatLabel('Model training complete')

        self.pipe = Pipeline([('xgb', xgboost.XGBClassifier(
            **xgb_dup,
            random_state = self.random_seed
        ))]).fit(X_dup, Y_dup)


        self.onnx = False


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

        # Limit to feature columns
        # feature_cols = [
        #     'BP_LEN','GENE_COUNT','DISEASE_COUNT','CENT_DIST','TEL_DIST', 
        #     'HI_SCORE','TS_SCORE','%HI','pLI','LOEUF','GC_CONTENT', 
        #     'POP_FREQ','CLINVAR_DENSITY','P_TRIPLO','P_HAPLO','gnomad_oe_lof',
        #     'gnomad_oe_lof_upper','gnomad_pLI','gnomad_lof_z','gnomad_mis_z',
        #     'gnomad_oe_mis_upper','episcore','sHet','exon_phastcons','hurles_hi',
        #     'rvis','promoter_cpg_count','eds','promoter_phastcons','cds_length',
        #     'gnomad_pNull'
        # ]

        feature_cols = [
            'BP_LEN','GENE_COUNT','DISEASE_COUNT','CENT_DIST','TEL_DIST', 
            'HI_SCORE','TS_SCORE','%HI','pLI','LOEUF','GC_CONTENT', 
            'POP_FREQ','CLINVAR_DENSITY'
        ]


        # Divide training data by CNV type
        XY = input.copy()

        if self.norm:
            with open(os.path.join(self.root_path, 'models', 'scalers.pickle'), 'rb') as f:
                col_scalers = pickle.load(f)
            for col in feature_cols:
                XY[col] = col_scalers[col].transform(XY[col].values.reshape(-1,1))

        XY_dup = XY[XY['CHANGE'] == 'DUP']
        XY_del = XY[XY['CHANGE'] == 'DEL']

        X_dup = XY_dup[feature_cols]
        X_del = XY_del[feature_cols]

        X_dup.columns = [f'f{i}' for i in range(len(X_dup.columns))]
        X_del.columns = [f'f{i}' for i in range(len(X_del.columns))]


        # Generate predictions
        bar_widgets[0] = progressbar.FormatLabel('Obtaining DEL predictions')
        bar.update(1)

        if self.onnx:
            
            sess = rt.InferenceSession(os.path.join(self.root_path, 'models', "del.onnx"))
            input_name = sess.get_inputs()[0].name
            del_pred = sess.run(None, {input_name: np.array(X_del, dtype = 'f')})
            del_pred = pd.DataFrame(del_pred[1]).rename(columns = {0:'BENIGN',1:'VUS',2:'PATHOGENIC'})
            
            bar_widgets[0] = progressbar.FormatLabel('Obtaining DUP predictions')
            bar.update(2)

            sess = rt.InferenceSession(os.path.join(self.root_path, 'models', "dup.onnx"))
            input_name = sess.get_inputs()[0].name
            dup_pred = sess.run(None, {input_name: np.array(X_dup, dtype = 'f')})
            dup_pred = pd.DataFrame(dup_pred[1]).rename(columns = {0:'BENIGN',1:'VUS',2:'PATHOGENIC'})

        else:
            del_pred = self.del_model.predict_proba(X_del)
            del_pred = pd.DataFrame(del_pred).rename(columns = {0:'BENIGN',1:'VUS',2:'PATHOGENIC'})

            bar_widgets[0] = progressbar.FormatLabel('Obtaining DUP predictions')
            bar.update(2)

            dup_pred = self.dup_model.predict_proba(X_dup)
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
    
    
    
        