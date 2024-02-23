import pandas as pd
import numpy as np
import sys
import os
import re
import math
import json
import subprocess
import sys
import progressbar
import vcf
# import pyranges as pr
from ..liftover.liftover import get_liftover_positions


class FeatureBuilder:

    def __init__(self, variant_df: pd.DataFrame, data_dir: str):

        print('Loading dependencies')

        # Set conda env
        self.conda_bin = os.path.join(sys.exec_prefix, 'bin')

        # Set variants
        self.variant_df = variant_df

        # Get root path
        self.data_dir = data_dir
        self.package_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.ref = os.path.join(self.data_dir, 'hg38.fa.gz')
        self.centromere_df = pd.read_csv(os.path.join(self.data_dir, 'centromeres.csv'), index_col = 0)
        self.telomere_df = pd.read_csv(os.path.join(self.data_dir, 'telomeres.csv'),  index_col = 0)
        self.gene_position_df = pd.read_csv(os.path.join(self.package_data_dir, 'gene_positions.csv')).rename(columns = {
            'Chromosomes':'CHROMOSOME',
            'GRCh38_start':'START',
            'GRCh38_stop':'END'
        })
        self.gene_position = {chrom: self.gene_position_df[self.gene_position_df['CHROMOSOME'] == chrom] for chrom in self.gene_position_df['CHROMOSOME'].unique()}

        # self.gene_position_ranges = pr.PyRanges(
        #         chromosomes=self.gene_position_df['CHROMOSOME'],
        #         starts=self.gene_position_df['START'],
        #         ends=self.gene_position_df['END']
        #     )

        # Replace nulls with -999 for genes with no gene associations
        self.mim2gene = pd.read_csv(os.path.join(self.data_dir, 'mim2gene_medgen'), sep = '\t')
        self.mim2gene.loc[self.mim2gene['GeneID'].str.contains('-'), 'GeneID'] = -999
        self.mim2gene = self.mim2gene.astype({'GeneID': int})
        self.mim2gene.columns = ['OMIM ID'] + self.mim2gene.columns[1:].to_list()

        self.gnomadSV_df = pd.read_csv(os.path.join(self.package_data_dir, 'gnomad4.csv.gz'), dtype = {'CHROM': str}).rename(columns = {'CHROM':'CHROMOSOME'})
        self.gnomadSV_df['CHROMOSOME'] = self.gnomadSV_df['CHROMOSOME'].str[3:]
        self.gnomadSV = {t: {chrom: self.gnomadSV_df[(self.gnomadSV_df['CHROMOSOME'] == chrom) & (self.gnomadSV_df['TYPE'] == t)] for chrom in self.gnomadSV_df['CHROMOSOME'].unique()} for t in ['DEL','DUP']}
        # self.gnomadSV_ranges = {t: {chrom: pr.PyRanges(
        #     chromosomes=self.gnomadSV[t][chrom]['CHROMOSOME'],
        #     starts=self.gnomadSV[t][chrom]['START'],
        #     ends=self.gnomadSV[t][chrom]['END']
        # ) for chrom in self.gnomadSV[t].keys()} for t in ['DEL','DUP']}


        self.hi_ts_df = pd.read_csv(os.path.join(self.data_dir, 'hi_ts_parsed.csv'))
        self.hi_ts = {chrom: self.hi_ts_df[self.hi_ts_df['CHROMOSOME'] == chrom] for chrom in self.hi_ts_df['CHROMOSOME'].unique()}
        # self.hi_ts_ranges = {chrom: pr.PyRanges(
        #     chromosomes=self.hi_ts[chrom][self.hi_ts[chrom]['CHROMOSOME'] == chrom]['CHROMOSOME'],  # Assuming there's a 'CHROMOSOME' column
        #     starts=self.hi_ts[chrom][self.hi_ts[chrom]['CHROMOSOME'] == chrom]['START'],
        #     ends=self.hi_ts[chrom][self.hi_ts[chrom]['CHROMOSOME'] == chrom]['END']
        # ) for chrom in self.hi_ts.keys()}

        # Create a breakpoint dataframe to determine chromosome arm length
        self.breakpoint_df = self.centromere_df.merge(self.telomere_df, on = 'CHROMOSOME', suffixes = ('_cen','_tel'))
        self.breakpoint_df['arm1_len'] = self.breakpoint_df['start_cen'] -  self.breakpoint_df['start_tel']
        self.breakpoint_df['arm2_len'] = self.breakpoint_df['end_tel'] - self.breakpoint_df['end_cen']

        # Conservation data
        # self.phylop_df = pd.read_csv('/igm/home/rsrxs003/rnb/output/BL-384/cons_score.csv', index_col=0)
        # self.phastcons_df = pd.read_csv('/igm/home/rsrxs003/rnb/output/BL-384/phastcons_score.csv', index_col=0)
        cols = ['bin', 'CHROMOSOME', 'minPos', 'maxPos', 'name', 'width', 'count', 'offset', 'wibName', 'lowerLimit', 'dataRange', 'validCount', 'sumData', 'sumSquares']
        cols_ = ['CHROMOSOME','minPos','maxPos','width','count','offset','lowerLimit','dataRange']
        col_types = {'minPos':int,'maxPos':int,'width':int,'count':int,'offset':int,'lowerLimit':float,'dataRange':float}

        # PhyloP
        self.phylop = pd.read_csv(os.path.join(self.data_dir,'phyloP100way.txt.gz'), sep = '\t')
        self.phylop.columns = cols
        self.phylop = self.phylop[cols_].astype(col_types)
        self.phylop['CHROMOSOME'] = self.phylop['CHROMOSOME'].str[3:]
        self.phylop = {chrom: self.phylop[self.phylop['CHROMOSOME'] == chrom] for chrom in self.phylop['CHROMOSOME'].unique()}
        with open(os.path.join(self.data_dir,'phyloP100way.wib'),'rb') as f:
            self.phylop_wib = f.read()

        # phastCons
        self.phastcons = pd.read_csv(os.path.join(self.data_dir,'phastCons100way.txt.gz'), sep = '\t')
        self.phastcons.columns = cols
        self.phastcons = self.phastcons[cols_].astype(col_types)
        self.phastcons['CHROMOSOME'] = self.phastcons['CHROMOSOME'].str[3:]
        self.phastcons = {chrom: self.phastcons[self.phastcons['CHROMOSOME'] == chrom] for chrom in self.phastcons['CHROMOSOME'].unique()}
        with open(os.path.join(self.data_dir,'phastCons100way.wib'),'rb') as f:
            self.phastcons_wib = f.read()

        # Null HI/TS res
        hi_ts_cols = ['NO_EVIDENCE','LITTLE_EVIDENCE','EMERGING_EVIDENCE',
                'SUFFICIENT_EVIDENCE','AUTOSOMAL_RECESSIVE','UNLIKELY',
                'NOT_EVALUATED']

        self.null_hi_ts = { **{
            '%HI': self.hi_ts_df['%HI'].max(),
            'pLI': self.hi_ts_df['pLI'].min(),
            'LOEUF': self.hi_ts_df['LOEUF'].max()
            },
            **{f"HI_{c}":0 for c in hi_ts_cols},
            **{f"TS_{c}":0 for c in hi_ts_cols}
        }


    def get_features(self):#, clinvar_ids: list): 
        """
        Build feature matrix that is compatible with sklearn / tensorflow for model 
        training. Features include:
            1. Number of genes spanned when gene is at least half spanned, partial when 
            less than half of gene is spanned
            2. Population frequency of CNV
            3. Copy Number
            4. Genome position
            5. Normalized distance to centromere
            6. Normalized distance to telemere
            7. CG content
            8. Haploinsufficient genomic region overlap (or triplosensitive)
            9. Length of CNV (in bp)
            10. GO terms (some sort of intersect function)
            11. Average dbNSFP
        """

        def get_bp_length(row):
            return row['END'] - row['START'] + 1
        
        def get_all_genes(row):
            """
            Query the gene position table to find the intersecting genes
            """        

            gene_position = self.gene_position[row['CHROMOSOME']]
            df = gene_position.loc[
                        (gene_position['START'].between(row['START'], row['END']))
                            |
                        (gene_position['END'].between(row['START'], row['END']))
                            |
                        ((gene_position['START'] <= row['START']) & (gene_position['END'] >= row['END']))
                ]    

            # row_interval = pr.PyRanges(
            #     chromosomes=[row['CHROMOSOME']],
            #     starts=[row['START']],
            #     ends=[row['END']]
            # )

            # intersect_df = self.gene_position.intersect(row_interval).df.rename(columns = {
            #     'Chromosome':'CHROMOSOME',
            #     'Start':'START',
            #     'End':'END'
            # })

            if df.empty:
                return []

            # intersect_df = intersect_df.merge(self.gene_position_df, on = ['CHROMOSOME','START','END'])

                
            return df['GeneID'].to_list()


        def get_omim_disease_genes(row):
            """
            Using the gene-disease OMIM table, find a list of genes that are 
            associated with a rare disease. Return the length of the resulting 
            gene list.
            """
            associated_diseases = self.mim2gene.loc[(self.mim2gene['GeneID'].isin(row["gene_info"])) & (self.mim2gene['type'] == 'phenotype')]
            return len(associated_diseases.index)


        def get_gene_count(row):
            """
            Count the total number of genes
            """
            return len(row["gene_info"])


        def get_centromere_distance(row):
            """
            Use the previously downloaded centromere coordinate data to obtain 
            the distance of the CNV to the centromere (normalized by chromosome 
            arm length)
            """
            cen_start, cen_end = self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'start_cen'], self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'end_cen']
            arm1_len, arm2_len = self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'arm1_len'], self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'arm2_len']
            if row['END'] < cen_start:
                return (cen_start - row['END']) / arm1_len
            else:
                return (row['START'] - cen_end) / arm2_len


        def get_telomere_distance(row):
            """
            Use the previously downloaded telomere coordinate data to obtain the 
            distance of the CNV to the telomere (normalized by chromosome arm 
            length)
            """
            tel_start, tel_end = self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'start_tel'], self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'end_tel']
            cen_start, cen_end = self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'start_cen'], self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'end_cen']
            arm1_len, arm2_len = self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'arm1_len'], self.breakpoint_df.loc[f"chr{row['CHROMOSOME']}", 'arm2_len']
            if row['END'] < cen_start:
                return (row['START'] - tel_start) / arm1_len
            else:
                return (tel_end - row['END']) / arm2_len


        def get_gc_content(row):
            command = f"{os.path.join(self.conda_bin,'samtools')} faidx {self.ref} chr{row['CHROMOSOME']}:{row['START']}-{row['END']}"
            p = subprocess.Popen(command,  stdout=subprocess.PIPE, shell = True)
            out, err = p.communicate()
            s = "".join(out.decode().strip().split()[1:])
            s = s.replace('N','') # Don't count N values
            cg_len = len(s.replace('A','').replace('T',''))
            try:
                return round(cg_len/len(s), 3)
            except:
                print(s)


        def get_hi_ts_regions(row):
            """
            Haploinsucciency(HS) and triplosensitive(TS) regions have been 
            downloaded by the dependency builder. This function reads the 
            donwloaded data set and determines if the CNV provided in "row"
            intersects with any HS or TS regions, and to what extent.
            """

            # def get_region_coverage(row, cnv_start, cnv_end):
            #     left = max(cnv_start, row['START'])
            #     right = min(cnv_end, row['END'])
            #     num = left - right
            #     denom = row['START'] - row['END']
            #     return num / denom

            hi_ts_df = self.hi_ts[row['CHROMOSOME']]

            # Filter to only include regions intersecting with CNV
            # df = self.hi_ts_df.loc[
            #     (self.hi_ts_df['CHROMOSOME'] == row['CHROMOSOME']) 
            #             & 
            #         (
            #             (self.hi_ts_df['START'].between(row['START'], row['END']))
            #                 |
            #             (self.hi_ts_df['START'].between(row['START'], row['END']))
            #         )
            #     ]

            # row_range = pr.PyRanges(
            #     chromosomes=[row['CHROMOSOME']],
            #     starts=[row['START']],
            #     ends=[row['END']]
            # )

            # intersect_df = hi_ts_ranges.intersect(row_range).df.rename(columns = {
            #     'Chromosome':'CHROMOSOME',
            #     'Start':'START',
            #     'End':'END'
            # })

            df = hi_ts_df.loc[
                        (hi_ts_df['START'].between(row['START'], row['END']))
                            |
                        (hi_ts_df['END'].between(row['START'], row['END']))
                            |
                        ((hi_ts_df['START'] <= row['START']) & (hi_ts_df['END'] >= row['END']))
                ]


            # Exit if the dataframe is empty
            hi_ts = ['HI','TS']
            hi_ts_cols = ['NO_EVIDENCE','LITTLE_EVIDENCE','EMERGING_EVIDENCE',
                'SUFFICIENT_EVIDENCE','AUTOSOMAL_RECESSIVE','UNLIKELY',
                'NOT_EVALUATED']


            if df.empty:
                return self.null_hi_ts

            # df = df.merge(hi_ts_df, on = ['CHROMOSOME','START','END'])

            # Return the score of the worst intersecting HI/TS region
            res = {
                '%HI': df['%HI'].min(),
                'pLI': df['pLI'].max(),
                'LOEUF': df['LOEUF'].min()
                }

            for ht in hi_ts:
                for col in hi_ts_cols:
                    res.update({f'{ht}_{col}': df[f'{ht}_{col}'].sum()})

            return res


        def get_population_frequency(row):
            """
            Gathers all gnomAD SV records that intersect with the CNV provided 
            in "row". Returns the average of the population frequencies observed 
            in intersecting variants
            """

            # def get_region_coverage(row, cnv_start, cnv_end):
            #     left = max(cnv_start, row['START'])
            #     right = min(cnv_end, row['stop'])
            #     num = left - right
            #     denom = row['start'] - row['stop']
            #     return num / denom

            # gnomad_ranges = self.gnomadSV_ranges[row['CHANGE']][row['CHROMOSOME']]
            gnomad_df = self.gnomadSV[row['CHANGE']][row['CHROMOSOME']]

            df = gnomad_df.loc[
                (gnomad_df['START'].between(row['START'], row['END']))
                    |
                (gnomad_df['END'].between(row['START'], row['END']))
                    |
                ((gnomad_df['START'] <= row['START']) & (gnomad_df['END'] >= row['END']))
            ]

            # gnomad_ranges = pr.PyRanges(
            #     chromosomes=gnomad_df['CHROMOSOME'],  # Assuming there's a 'CHROMOSOME' column
            #     starts=gnomad_df['START'],
            #     ends=gnomad_df['END']
            # )

            # row_range = pr.PyRanges(
            #     chromosomes=[row['CHROMOSOME']],
            #     starts=[row['START']],
            #     ends=[row['END']]
            # )

            # intersect_df = gnomad_ranges.intersect(row_range).df.rename(columns = {
            #     'Chromosome':'CHROMOSOME',
            #     'Start':'START',
            #     'End':'END'
            # })

            if df.empty:
                return 0

            # gnomad_df = intersect_df.merge(self.gnomadSV[row['CHANGE']][row['CHROMOSOME']], on = ['CHROMOSOME','START','END'])
            
            df['percent_coverage'] = (
            df[['START', 'END']].clip(lower=row['START'], upper=row['END']).diff(axis=1) / 
                        df[['START', 'END']].diff(axis=1)
                        )['END']

            # Only consider variants that cover at least half of the CNV
            df = df.loc[df['percent_coverage'] >= 0.5]

            # return gnomad_df['pop_freq'].mean()
            return df['POP_FREQ'].max()

        
        def get_clinvar_path_density(row, clinvar_reader):
            path_count = 0
            query = clinvar_reader.fetch(chrom = row['CHROMOSOME'], start = row['START'], end = row['END'])
            for q in query:
                try:
                    if re.search('PATHOGENIC', q.INFO['CLNSIG'][0].upper()):
                        path_count += 1
                except:
                    pass
            return path_count


        def moving_average(values, window_size):
            """
            Calculate the moving average using a specified window size.

            Parameters:
            values (array-like): The input data for which the moving average is computed.
            window_size (int): The size of the moving window.

            Returns:
            array-like: The moving average of the input data.
            """
            if window_size < 1:
                raise ValueError("Window size must be at least 1")

            # Create a window: using a uniform weight (simple moving average)
            window = np.ones(int(window_size)) / float(window_size)

            # Compute the moving average using convolution
            moving_avg = np.convolve(values, window, 'valid')

            return moving_avg


        def get_wib_values(cons_df, wib_data):

            chroms, starts, steps, values = [], [], [], []

            for idx, row in cons_df.iterrows():

                for i in range(row['count']):
                    byte_value = wib_data[row['offset'] + i]
                    if byte_value < 128:
                        value = row['lowerLimit'] + (row['dataRange'] * (byte_value / 127.0))
                        chroms.append(row['CHROMOSOME'])
                        starts.append(row['minPos'] + i * row['width'])
                        steps.append(row['width'])
                        values.append(value)

            return pd.DataFrame({
                'chrom': chroms,
                'start': starts,
                'step': steps,
                'value': values
            })


        def get_cons(row, window_size, cons, wib):

            # Get necessary rows from wiggle
            cons_df = cons[row['CHROMOSOME']]
            cons_df = cons_df[
                (row['START'] - window_size <= cons_df['maxPos'] )
                    &
                (cons_df['minPos'] <= row['END'] + window_size)
            ]

            # Get wib values
            wib_values = get_wib_values(cons_df, wib)
            wib_values = wib_values[['start','value']].rename(columns = 
                                                    {'start':'POS','value':'VALUE'})
            wib_values = wib_values[wib_values['POS'].between(
                row['START'] - (window_size / 2), row['END'] + (window_size / 2))]

            # Get rolling average
            mov_avg = moving_average(wib_values['VALUE'], window_size)

            # Take maximum
            return mov_avg.max().round(5)
    


        def get_ptriplo(row, c_data):
            try:
                max_ = c_data[c_data['geneID'].isin(row['gene_info'])]['pHaplo'].max()
                return max_ if not pd.isna(max_) else 0
            except:
                return 0
        

        def get_htriplo(row, c_data):
            try:
                max_ = c_data[c_data['geneID'].isin(row['gene_info'])]['pTriplo'].max()
                return max_ if not pd.isna(max_) else 0
            except:
                return 0


        def get_collins_data(row, c_data_full, col, func_):
            try:
                if func_ == 'max':
                    max_ = c_data_full[c_data_full['geneID'].isin(row['gene_info'])][col].max()
                    return max_ if not pd.isna(max_) else 0
                elif func_ == 'min':
                    min_ = c_data_full[c_data_full['geneID'].isin(row['gene_info'])][col].min()
                    return min_ if not pd.isna(min_) else 1
                elif func_ == 'mean':
                    mean_ = c_data_full[c_data_full['geneID'].isin(row['gene_info'])][col].mean()
                    return mean_ if not pd.isna(mean_) else 0.5
            except:
                if func_ == 'max':
                    return 0
                elif func_ == 'min':
                    return 1
                else:
                    return 0

        cnv_df = self.variant_df.copy()

        # Intialize progress bar
        bar_widgets = [progressbar.FormatLabel('Building Features'), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.Timer()]
        bar = progressbar.ProgressBar(maxval=12, \
            widgets=bar_widgets)
        bar_widgets[0] = progressbar.FormatLabel('Ensuring positions are in hg38')
        bar.start()

        # Map to hg38
        cnv_df['build'] = 'GRCh38'
        cnv_df['variant'] = cnv_df.apply(get_liftover_positions, axis = 1)
        cnv_df = pd.concat([cnv_df.drop(['variant','START','END'], axis = 1), cnv_df['variant'].apply(pd.Series)], axis = 1)
        cnv_df = cnv_df.astype({'START': 'int32', 'END': 'int32'})
        original_cols = list(cnv_df.columns)

        # Get features
        bar_widgets[0] = progressbar.FormatLabel('Collecting CNV length')
        bar.update(1)
        cnv_df['BP_LEN'] = cnv_df.apply(get_bp_length, axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Collecting gene data')
        bar.update(2)
        cnv_df['gene_info'] = cnv_df.apply(get_all_genes, axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Calculating gene count')
        bar.update(3)
        cnv_df['GENE_COUNT'] = cnv_df.apply(get_gene_count, axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Calculating OMIM disease count')
        bar.update(4)
        cnv_df['DISEASE_COUNT'] = cnv_df.apply(get_omim_disease_genes, axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Calculating centromere distance count')
        bar.update(5)
        cnv_df['CENT_DIST'] = cnv_df.apply(get_centromere_distance, axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Calculating HI/TS coverage')
        bar.update(6)
        cnv_df['hi_ts_region_scores'] = cnv_df.apply(get_hi_ts_regions, axis = 1)
        cnv_df = pd.concat([cnv_df.drop('hi_ts_region_scores', axis = 1), cnv_df['hi_ts_region_scores'].apply(pd.Series) ], axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Calculating GC content')
        bar.update(7)
        cnv_df['GC_CONTENT'] = cnv_df.apply(get_gc_content, axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Collecting population frequencies')
        bar.update(8)
        cnv_df['POP_FREQ'] = cnv_df.apply(get_population_frequency, axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('Collecting ClinVar short variant density')
        bar.update(9)

        # Create reader of ClinVar short variants
        clinvar_path = os.path.join(self.data_dir, 'clinvar.vcf.gz')
        clinvar_reader = vcf.Reader(filename = clinvar_path, compressed=True, encoding='ISO-8859-1')
        cnv_df['CLINVAR_DENSITY'] = cnv_df.apply(get_clinvar_path_density, clinvar_reader = clinvar_reader, axis = 1)
        
        bar_widgets[0] = progressbar.FormatLabel('ClinVar density generated')
        bar.update(10)
        cnv_df['PHYLOP_SCORE'] = cnv_df.apply(
            get_cons, 
            window_size = 200,
            cons = self.phylop,
            wib = self.phylop_wib,
            axis = 1)
        cnv_df['PHASTCONS_SCORE'] = cnv_df.apply(
            get_cons, 
            window_size = 5000,
            cons = self.phastcons,
            wib = self.phastcons_wib,
            axis = 1)

        bar_widgets[0] = progressbar.FormatLabel('All features generated, normalizing')
        bar.update(11)

        # Fill NAs
        cnv_df = cnv_df.fillna({
            'GC_CONTENT': cnv_df['GC_CONTENT'].median(),
            'CLINVAR_DENSITY': 0,
            'POP_FREQ': 0,
            'PHYLOP_SCORE': cnv_df['PHYLOP_SCORE'].min(),
            'PHASTCONS_SCORE': cnv_df['PHASTCONS_SCORE'].min()
        })

        cols_to_normalize = [
            'BP_LEN','GENE_COUNT','DISEASE_COUNT','CENT_DIST','%HI','pLI',
            'LOEUF','GC_CONTENT','POP_FREQ','CLINVAR_DENSITY','PHYLOP_SCORE',
            'PHASTCONS_SCORE']

        hs_ts_cols = []
        hi_ts = ['HI','TS']
        hi_ts_cols = ['NO_EVIDENCE','LITTLE_EVIDENCE','EMERGING_EVIDENCE',
                'SUFFICIENT_EVIDENCE','AUTOSOMAL_RECESSIVE','UNLIKELY',
                'NOT_EVALUATED']

        for ht in hi_ts:
            for col in hi_ts_cols:
                hs_ts_cols.append(f"{ht}_{col}")
                cnv_df = cnv_df.astype({f"{ht}_{col}":int})

        feature_cols = cols_to_normalize + hs_ts_cols
        
        for col in feature_cols:
            cnv_df.loc[:,col] = cnv_df[col].astype(float)             


        bar_widgets[0] = progressbar.FormatLabel('Feature generation complete')
        bar.update(12)
        bar.finish()

        cols = ['CHROMOSOME','START','END','CHANGE'] 
        cols = cols + [oc for oc in original_cols if oc not in cols] + ['gene_info'] + cols_to_normalize + hs_ts_cols
        cols = [c for c in cols if c not in ['build','HI_SCORE','TS_SCORE']]

        self.feature_df = cnv_df[cols]
        
