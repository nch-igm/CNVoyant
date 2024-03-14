import os
import PIL
import re
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import sys
from PIL import Image
from CNVoyant import FeatureBuilder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


class Explainer:

    def __init__(self, cnv_coordinates, output_dir, classifier):

        # Get root path
        self.chrom = cnv_coordinates['CHROMOSOME']
        self.start = cnv_coordinates['START']
        self.end = cnv_coordinates['END']
        self.var_type = cnv_coordinates['CHANGE']
        self.output_dir = output_dir
        self.classifier = classifier
        self.data_dir = self.classifier.data_dir
        self.cl = self.classifier.del_model if self.var_type == 'DEL' else self.classifier.dup_model


        # Set variable labels
        self.readable_names = {
            'BP_LEN':'BP Length',
            'GENE_COUNT': 'Gene Count',
            'DISEASE_COUNT':'Disease Count',
            'EXON_COUNT':'Exon Count',
            'REG_COUNT':'Reg Count',
            'CENT_DIST': 'Cent. Distance',
            'TEL_DIST': 'Tel. Distance',
            '%HI':'HI Index',
            'pLI':'pLI',
            'LOEUF':'LOEUF',
            'GC_CONTENT':'GC Content',
            'POP_FREQ':'Pop. Frequency',
            'CLINVAR_DENSITY':'ClinVar SNV/indel Density',
            'PHYLOP_SCORE':'phyloP',
            'PHASTCONS_SCORE':'PhastCons',
            'HI_NO_EVIDENCE':'HI (No Evidence)',
            'HI_LITTLE_EVIDENCE':'HI (Little Evidence)',
            'HI_EMERGING_EVIDENCE':'HI (Emerging Evidence)',
            'HI_SUFFICIENT_EVIDENCE':'HI (Sufficient Evidence)',
            'HI_AUTOSOMAL_RECESSIVE':'HI (Autosomal Recessive)',
            'HI_UNLIKELY':'HI (Unlikely)',
            'HI_NOT_EVALUATED':'HI (Not Evaluated)',
            'TS_NO_EVIDENCE':'TS (No Evidence)',
            'TS_LITTLE_EVIDENCE':'TS (Little Evidence)',
            'TS_EMERGING_EVIDENCE':'TS (Emerging Evidence)',
            'TS_SUFFICIENT_EVIDENCE':'TS (Sufficient Evidence)',
            'TS_AUTOSOMAL_RECESSIVE':'TS (Autosomal Recessive)',
            'TS_UNLIKELY':'TS (Unlikely)',
            'TS_NOT_EVALUATED':'TS (Not Evaluated)'
        }

        # Define feature columns
        hs_ts_cols = []
        hi_ts = ['HI','TS']
        hi_ts_cols = ['NO_EVIDENCE','LITTLE_EVIDENCE','EMERGING_EVIDENCE',
                'SUFFICIENT_EVIDENCE','AUTOSOMAL_RECESSIVE','UNLIKELY','NOT_EVALUATED']

        for ht in hi_ts:
            for col in hi_ts_cols:
                hs_ts_cols.append(f"{ht}_{col}")

        self.feature_cols = [
                'BP_LEN','GENE_COUNT','DISEASE_COUNT','EXON_COUNT','REG_COUNT',
                'CENT_DIST','TEL_DIST','%HI','pLI', 'LOEUF','GC_CONTENT','POP_FREQ',
                'CLINVAR_DENSITY','PHYLOP_SCORE','PHASTCONS_SCORE'] + hs_ts_cols


    def validate_input(self, variant_df):
        """
        Validates the input variant DataFrame.

        Args:
            variant_df (pd.DataFrame): The input variant DataFrame to be validated.

        Returns:
            pd.DataFrame: The validated variant DataFrame with only the valid columns.

        Raises:
            ValueError: If the inputted variants are not a Series or if they do not contain the required columns.
        """
        # Validate input data type
        if not isinstance(variant_df, pd.DataFrame):
            raise ValueError("Inputted variants must be a DataFrame")

        # Validate input columns
        valid_columns = ["CHROMOSOME", "START", "END", "CHANGE"]
        if not set(valid_columns).issubset(variant_df.columns):
            raise ValueError("Inputted variant must contain 'CHROMOSOME', 'START', 'END', and 'CHANGE' columns")


        # Ensure variant is prefixed with 'chr'
        variant_df['CHROMOSOME'] = variant_df['CHROMOSOME'].astype(str).apply(lambda x: x if x.startswith('chr') else f'chr{x}')

        return variant_df[valid_columns]


    def normalize_input(self):
        """
        Normalizes the input data using the provided classifier scalar.
        """

        XY = self.feature_builder.feature_df.copy()

        # Log transform large columns
        log_null_value = 0.1

        XY['BP_LEN'] = np.log(XY['BP_LEN'])
        XY['GENE_COUNT'] = np.log(XY['GENE_COUNT'].replace(0,log_null_value))
        XY['DISEASE_COUNT'] = np.log(XY['DISEASE_COUNT'].replace(0,log_null_value))
        XY['CLINVAR_DENSITY'] = np.log(XY['CLINVAR_DENSITY'].replace(0,log_null_value))
        
        # Apply scalars
        for col in self.feature_cols:
            XY[col] = self.classifier.col_scalers[col].transform(XY[col].values.reshape(-1, 1))

        return XY


    def plot_shap_force_plot(self):
        """
        Plots a SHAP force plot for a single prediction.
        """

        # Convert the Pandas Series to a DataFrame because SHAP expects 2D input
        data_frame = self.norm_variant_df[self.feature_cols]#.to_frame().T.astype(float).round(2)
        val_df = self.feature_builder.feature_df[self.feature_cols].copy()
        int_cols = [c for c in self.feature_cols if re.search('COUNT',c) or re.search('_DIST',c) or re.search('HI_',c) or re.search('TS_',c)] + ['BP_LEN','CLINVAR_DENSITY']
        float_cols = [c for c in self.feature_cols if c not in int_cols]

        for c in int_cols:
            val_df[c] = val_df[c].astype(int)
            val_df[c] = val_df[c].astype(str)

        for c in float_cols:
            val_df[c] = val_df[c].round(3)
        
        # Ensure the DataFrame has the correct column names
        val_df.columns = [self.readable_names[fn] for fn in val_df.columns]
        
        # Initialize the SHAP explainer
        explainer = shap.TreeExplainer(self.cl)
        
        # Calculate SHAP values for the input data
        shap_values = explainer.shap_values(data_frame)

        label_dict = {
            0:'Benign',
            1:'VUS',
            2:'Pathogenic'
        }

        for k,v in label_dict.items():
            shap.force_plot(
                explainer.expected_value[k],
                shap_values[k],
                val_df.iloc[0],
                matplotlib=True,
                text_rotation=15,
                figsize=(15,5),
                show=False
            )
        
            plt.title(v,fontdict={'fontsize':30,'fontweight':'bold'}, pad = 120)
            plt.xticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,f"force_plot_{v}.png"), dpi = 500)
            plt.close()


    def combine_images(self, filenames, output_filename):
        """
        Combines several images into one vertically.

        Parameters:
        - filenames: List of filenames of the images to combine.
        - output_filename: The filename for the combined image.
        """
        images = [Image.open(fn) for fn in filenames]
        widths, heights = zip(*(i.size for i in images))

        total_height = sum(heights)
        max_width = max(widths)

        new_im = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        new_im.save(output_filename)


    def explain(self):

        # Validate inputted variants
        self.variant = pd.Series({'CHROMOSOME': self.chrom, 'START': self.start, 'END': self.end, 'CHANGE': self.var_type})
        self.variant_df = pd.DataFrame(self.variant).T
        self.variant_df = self.validate_input(self.variant_df)

        # Get feature values
        self.feature_builder = FeatureBuilder(self.variant_df, self.data_dir)
        self.feature_builder.get_features()

        # Normalize features
        self.norm_variant_df = self.normalize_input()

        # Force plot for a specific class (e.g., the predicted class)
        self.plot_shap_force_plot()
        self.combine_images(
            [os.path.join(self.output_dir,f"force_plot_{v}.png") for v in ['Benign','VUS','Pathogenic']], 
            os.path.join(self.output_dir,'combined_force_plot.png')
            )

        print(f"SHAP force plot saved to {os.path.join(self.output_dir,'combined_force_plot.png')}")
