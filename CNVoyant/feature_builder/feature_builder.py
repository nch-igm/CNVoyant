import pandas as pd
import os
import re
import math
import json
import subprocess
import sys
import progressbar
import vcf
from sklearn.preprocessing import MinMaxScaler
from ..liftover.liftover import get_liftover_positions


class FeatureBuilder:

    def __init__(self, variant_df: pd.DataFrame, data_dir: str, ref: str):

        # Get root path
        self.data_dir = data_dir
        self.ref = ref
        self.centromere_df = pd.read_csv(os.path.join(self.data_dir, 'centromeres.csv'), index_col = 0)
        self.telomere_df = pd.read_csv(os.path.join(self.data_dir, 'telomeres.csv'),  index_col = 0)

        # Replace nulls with -999 for genes with no gene associations
        self.mim2gene = pd.read_csv(os.path.join(self.data_dir, 'mim2gene_medgen'), sep = '\t')
        self.mim2gene.loc[self.mim2gene['GeneID'].str.contains('-'), 'GeneID'] = -999
        self.mim2gene = self.mim2gene.astype({'GeneID': int})
        self.mim2gene.columns = ['OMIM ID'] + self.mim2gene.columns[1:].to_list()

        self.gnomadSV = pd.read_csv(os.path.join(self.data_dir, 'gnomad_frequencies.csv'), dtype = {'chrom': str})
        self.hi_ts_df = pd.read_csv(os.path.join(self.data_dir, 'hi_ts_regions.csv'))
        self.hi_ts_mapped_df = pd.read_csv(os.path.join(self.data_dir, 'hi_ts_map.csv'))
        self.variant_df = variant_df

        # Create a breakpoint dataframe to determine chromosome arm length
        self.breakpoint_df = self.centromere_df.merge(self.telomere_df, on = 'CHROMOSOME', suffixes = ('_cen','_tel'))
        self.breakpoint_df['arm1_len'] = self.breakpoint_df['start_cen'] -  self.breakpoint_df['start_tel']
        self.breakpoint_df['arm2_len'] = self.breakpoint_df['end_tel'] - self.breakpoint_df['end_cen']


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
        
        def get_all_genes(row, gene_position_df):
            """
            Query the gene position table to find the intersecting genes
            """

            intersect_df = gene_position_df[
                (row['CHROMOSOME'] == gene_position_df['Chromosomes'])
                    &
                (
                    (
                        (row['START'] <= gene_position_df['GRCh38_start'])
                            &
                        (row['END'] >= gene_position_df['GRCh38_start'])
                    )
                        |
                    (
                        (row['START'] <= gene_position_df['GRCh38_stop'])
                            &
                        (row['END'] >= gene_position_df['GRCh38_stop'])
                    )
                        |
                    (
                        (row['START'] >= gene_position_df['GRCh38_start'])
                            &
                        (row['END'] <= gene_position_df['GRCh38_stop'])
                    )
                )
            ]
                
            return intersect_df['GeneID'].to_list()


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
            command = f"samtools faidx {self.ref} chr{row['CHROMOSOME']}:{row['START']}-{row['END']}"
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

            def get_region_coverage(row, cnv_start, cnv_end):
                left = max(cnv_start, row['START'])
                right = min(cnv_end, row['END'])
                num = left - right
                denom = row['START'] - row['END']
                return num / denom

            def get_hi_ts_scores(row):
                data = {}
                data['HI_SCORE'] = self.hi_ts_mapped_df.loc[self.hi_ts_mapped_df['HI_TS_Score'] == row['HI Score']].reset_index().loc[0, 'Mapped_Score']
                data['TS_SCORE'] = self.hi_ts_mapped_df.loc[self.hi_ts_mapped_df['HI_TS_Score'] == row['TS Score']].reset_index().loc[0, 'Mapped_Score']
                return data

            # Filter to only include regions intersecting with CNV
            df = self.hi_ts_df.loc[
                (self.hi_ts_df['CHROMOSOME'] == row['CHROMOSOME']) 
                        & 
                    (
                        (self.hi_ts_df['START'].between(row['START'], row['END']))
                            |
                        (self.hi_ts_df['START'].between(row['START'], row['END']))
                    )
                ]

            # Only keep the gene entries
            # print(row['clinvar_id'], row['gene_count'], f'chr{row["chrom"]}', row['GRCh38_start'], row['GRCh38_end'], len(df.index), len(df[df['type'] != 'R'].index), len(df[(df['%HI'] != '‐') | (df['pLI'] != '‐') | (df['LOEUF'] != '‐')].index), row['gene_info'])
            df = df[(df['%HI'] != '‐') & (df['pLI'] != '‐') & (df['LOEUF'] != '‐')]

            # Exit if the dataframe is empty
            if df.empty:
                return {
                    'HI_SCORE': 0,
                    'TS_SCORE': 0,
                    '%HI': 100,
                    'pLI': 0,
                    'LOEUF': 2 # max value in HI table
                    }

            # Get coverage percentage for each intersecting region
            df['percent_coverage'] = df.apply(
                get_region_coverage, 
                cnv_start = row['START'], 
                cnv_end = row['END'],
                axis = 1)

            # Map HI and TS scores from pre-defined values
            df['data'] = df.apply(get_hi_ts_scores, axis = 1)
            df = pd.concat([df.drop(['HI Score', 'TS Score', 'data'], axis = 1), df['data'].apply(pd.Series)], axis = 1)

            # Calculate final overlap * (TS/HS)
            df['adjusted_HI_score'] = df['percent_coverage'] * df['HI_SCORE']
            df['adjusted_TS_score'] = df['percent_coverage'] * df['TS_SCORE']

            # Fill null values in float columns
            for col in ['%HI', 'pLI', 'LOEUF']:
                try:
                    df[col] = df[col].astype(float)
                except Exception as e:
                    print(type(e), e)
                    print(df)

            df['adjusted_%HI'] = df['percent_coverage'] * df['%HI']
            df['adjusted_pLI'] = df['percent_coverage'] * df['pLI']
            df['adjusted_LOEUF'] = df['percent_coverage'] * df['LOEUF']

            # Return the score of the worst intersecting HI/TS region
            return {
                'HI_SCORE': df['adjusted_HI_score'].max(),
                'TS_SCORE': df['adjusted_TS_score'].max(),
                '%HI': df['adjusted_%HI'].min(),
                'pLI': df['adjusted_pLI'].max(),
                'LOEUF': df['adjusted_LOEUF'].min()
                }


        def get_population_frequency(row):
            """
            Gathers all gnomAD SV records that intersect with the CNV provided 
            in "row". Returns the average of the population frequencies observed 
            in intersecting variants
            """

            def get_region_coverage(row, cnv_start, cnv_end):
                left = max(cnv_start, row['start'])
                right = min(cnv_end, row['stop'])
                num = left - right
                denom = row['start'] - row['stop']
                return num / denom

            gnomad_df = self.gnomadSV.loc[
                    (self.gnomadSV['start'].between(row['START'], row['END'])) 
                        | 
                    (self.gnomadSV['stop'].between(row['START'], row['END']))
                ]

            if not gnomad_df.empty:
                gnomad_df['percent_coverage'] = gnomad_df.apply(
                    get_region_coverage, 
                    cnv_start = row['START'], 
                    cnv_end = row['END'],
                    axis = 1)

            else:
                gnomad_df['percent_coverage'] = 0

            # Only consider variants that cover at least half of the CNV
            gnomad_df = gnomad_df.loc[gnomad_df['percent_coverage'] >= 0.5]

            return gnomad_df['pop_freq'].mean()

        
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

        # Load dependencies
        gene_position_df = pd.read_csv(os.path.join(self.data_dir, 'gene_positions.csv'))

        # Intialize progress bar
        bar_widgets = [progressbar.FormatLabel('Building Features'), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.Timer()]
        bar = progressbar.ProgressBar(maxval=14, \
            widgets=bar_widgets)
        bar.start()
        bar_widgets[0] = progressbar.FormatLabel('Ensuring positions are in hg38')

        # Map to hg38
        # cnv_df['build'] = config['build']
        cnv_df['build'] = 'GRCh38'
        cnv_df['variant'] = cnv_df.apply(get_liftover_positions, axis = 1)
        cnv_df = pd.concat([cnv_df.drop(['variant','START','END'], axis = 1), cnv_df['variant'].apply(pd.Series)], axis = 1)
        cnv_df = cnv_df.astype({'START': 'int32', 'END': 'int32'})
        original_cols = list(cnv_df.columns)
        # cnv_df = cnv_df[['CHROMOSOME','START','END','CHANGE']]

        # Get features
        bar.update(1)
        bar_widgets[0] = progressbar.FormatLabel('Collecting CNV length')
        cnv_df['BP_LEN'] = cnv_df.apply(get_bp_length, axis = 1)

        bar.update(2)
        bar_widgets[0] = progressbar.FormatLabel('Collecting gene data')
        cnv_df['gene_info'] = cnv_df.apply(get_all_genes, gene_position_df = gene_position_df, axis = 1)

        bar.update(3)
        bar_widgets[0] = progressbar.FormatLabel('Calculating gene count')
        cnv_df['GENE_COUNT'] = cnv_df.apply(get_gene_count, axis = 1)

        bar.update(4)
        bar_widgets[0] = progressbar.FormatLabel('Calculating OMIM disease count')
        cnv_df['DISEASE_COUNT'] = cnv_df.apply(get_omim_disease_genes, axis = 1)

        bar.update(5)
        bar_widgets[0] = progressbar.FormatLabel('Calculating centromere distance count')
        cnv_df['CENT_DIST'] = cnv_df.apply(get_centromere_distance, axis = 1)

        bar.update(6)
        bar_widgets[0] = progressbar.FormatLabel('Calculating telomere distance count')
        cnv_df['TEL_DIST'] = cnv_df.apply(get_telomere_distance, axis = 1)

        bar.update(7)
        bar_widgets[0] = progressbar.FormatLabel('Calculating HI/TS coverage')
        cnv_df['hi_ts_region_scores'] = cnv_df.apply(get_hi_ts_regions, axis = 1)
        cnv_df = pd.concat([cnv_df.drop('hi_ts_region_scores', axis = 1), cnv_df['hi_ts_region_scores'].apply(pd.Series) ], axis = 1)

        bar.update(8)
        bar_widgets[0] = progressbar.FormatLabel('Calculating GC content')
        cnv_df['GC_CONTENT'] = cnv_df.apply(get_gc_content, axis = 1)

        bar.update(9)
        bar_widgets[0] = progressbar.FormatLabel('Collecting population frequencies')
        cnv_df['POP_FREQ'] = cnv_df.apply(get_population_frequency, axis = 1).fillna(0)

        bar.update(10)
        bar_widgets[0] = progressbar.FormatLabel('Collecting ClinVar short variant density')

        # Create reader of ClinVar short variants
        clinvar_path = os.path.join(self.data_dir, 'clinvar.vcf.gz')
        clinvar_reader = vcf.Reader(filename = clinvar_path, compressed=True, encoding='ISO-8859-1')
        cnv_df['CLINVAR_DENSITY'] = cnv_df.apply(get_clinvar_path_density, clinvar_reader = clinvar_reader, axis = 1)
        
        # bar.update(11)
        # bar_widgets[0] = progressbar.FormatLabel('ClinVar density generated')

        # # Get maximum haploinsuffeciency and triplosensitivity probabilities 
        # collins_path = os.path.join(self.data_dir, 'TS_score_geneid.csv')
        # collins_df = pd.read_csv(collins_path)
        # cnv_df['P_TRIPLO'] = cnv_df.apply(get_ptriplo, c_data = collins_df, axis = 1)
        # cnv_df['P_HAPLO'] = cnv_df.apply(get_htriplo, c_data = collins_df, axis = 1)

        # bar.update(12)
        # bar_widgets[0] = progressbar.FormatLabel('Probability HI/TS generated')

        # # Get other collins features
        # collins_full_path = os.path.join(self.data_dir, 'collins_raw_geneid.csv')
        # collins_full_df = pd.read_csv(collins_full_path)
        # cnv_df['gnomad_oe_lof'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'gnomad_oe_lof', func_ = 'min', axis = 1)
        # cnv_df['gnomad_oe_lof_upper'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'gnomad_oe_lof_upper', func_ = 'min', axis = 1)
        # cnv_df['gnomad_pLI'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'gnomad_pLI', func_ = 'max', axis = 1)
        # cnv_df['gnomad_lof_z'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'gnomad_lof_z', func_ = 'max', axis = 1)
        # cnv_df['gnomad_mis_z'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'gnomad_mis_z', func_ = 'max', axis = 1)
        # cnv_df['gnomad_oe_mis_upper'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'gnomad_oe_mis_upper', func_ = 'min', axis = 1)
        # cnv_df['episcore'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'episcore', func_ = 'max', axis = 1)
        # cnv_df['sHet'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'sHet', func_ = 'max', axis = 1)
        # cnv_df['exon_phastcons'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'exon_phastcons', func_ = 'max', axis = 1)
        # cnv_df['hurles_hi'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'hurles_hi', func_ = 'max', axis = 1)
        # cnv_df['rvis'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'rvis', func_ = 'max', axis = 1)
        # cnv_df['promoter_cpg_count'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'promoter_cpg_count', func_ = 'max', axis = 1)
        # cnv_df['eds'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'eds', func_ = 'max', axis = 1)
        # cnv_df['promoter_phastcons'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'promoter_phastcons', func_ = 'max', axis = 1)
        # cnv_df['cds_length'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'cds_length', func_ = 'max', axis = 1)
        # cnv_df['gnomad_pNull'] = cnv_df.apply(get_collins_data, c_data_full = collins_full_df, col = 'gnomad_pNull', func_ = 'min', axis = 1)
        


        bar.update(11)
        # bar.update(13)
        bar_widgets[0] = progressbar.FormatLabel('All features generated, normalizing')


        # Fill NAs
        cnv_df = cnv_df.fillna({
            'GC_CONTENT': .5,
            'CLINVAR_DENSITY': 0,
            'POP_FREQ': 0
        })


        # Normalize features
        cnv_norm_df = cnv_df.copy()
        scaler = MinMaxScaler()

        cols_to_normalize = [
            'BP_LEN','GENE_COUNT','DISEASE_COUNT','CENT_DIST','TEL_DIST', 
            'HI_SCORE','TS_SCORE','%HI','pLI','LOEUF','GC_CONTENT', 
            'POP_FREQ','CLINVAR_DENSITY'
        ]
        # cols_to_normalize = [
        #     'BP_LEN','GENE_COUNT','DISEASE_COUNT','CENT_DIST','TEL_DIST', 
        #     'HI_SCORE','TS_SCORE','%HI','pLI','LOEUF','GC_CONTENT', 
        #     'POP_FREQ','CLINVAR_DENSITY','P_TRIPLO','P_HAPLO','gnomad_oe_lof',
        #     'gnomad_oe_lof_upper','gnomad_pLI','gnomad_lof_z','gnomad_mis_z',
        #     'gnomad_oe_mis_upper','episcore','sHet','exon_phastcons','hurles_hi',
        #     'rvis','promoter_cpg_count','eds','promoter_phastcons','cds_length',
        #     'gnomad_pNull'
        # ]

        # for col in cols_to_normalize:
        #     cnv_norm_df[col] = scaler.fit_transform(cnv_norm_df[col].values.reshape(-1,1))

        bar_widgets[0] = progressbar.FormatLabel('Feature generation complete')
        bar.update(12)
        # bar.update(14)
        bar.finish()

        cols = ['CHROMOSOME','START','END','CHANGE'] 
        cols = cols + [oc for oc in original_cols if oc not in cols] + ['gene_info'] + cols_to_normalize
        cols = [c for c in cols if c != 'build']

        # self.norm_feature_df, self.feature_df = cnv_norm_df[cols], cnv_df[cols]
        self.feature_df = cnv_df[cols]
        
