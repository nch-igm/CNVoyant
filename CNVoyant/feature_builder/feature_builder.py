import pandas as pd
import numpy as np
import sys
import os
import re
import shutil
import shlex
import math
import json
import subprocess
import sys
import progressbar
import vcf
import pyBigWig
import pybedtools
from ..liftover.liftover import get_liftover_positions

# Set pandas options
try:
    pd.set_option('future.no_silent_downcasting', True)
except:
    pass


class FeatureBuilder:

    def __init__(self, variant_df: pd.DataFrame, data_dir: str):


        def worker(cmd):
            """
            Runs a bash command using subprocess module
            """
            try:
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                output = e.output
            return output.decode('utf-8')

        print('Loading dependencies')

        # Set conda env
        self.conda_bin = os.path.join(sys.exec_prefix, 'bin')

        # Set variants
        pd.Series(list(range(1,23)) + ['X','Y'])
        self.variant_df = variant_df.sort_values(['CHROMOSOME','START'])

        # Get root path
        self.data_dir = data_dir
        self.temp_data_dir = os.path.join(self.data_dir,'temp')
        if not os.path.exists(self.temp_data_dir):
            os.mkdir(self.temp_data_dir)
        self.package_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        # Output variants for limiting databases
        self.vars_bed = os.path.join(self.temp_data_dir,'vars.bed')
        self.variant_df['CHROMOSOME'] = self.variant_df['CHROMOSOME'].apply(lambda x: x[3:] if re.search('chr',x) else x)
        self.variant_df[['CHROMOSOME','START','END']].drop_duplicates().to_csv(self.vars_bed, sep = '\t', index = False, header = None)
        
        self.vars_chr_bed = os.path.join(self.temp_data_dir,'vars_chr.bed')
        self.variant_df['CHROMOSOME'] = self.variant_df['CHROMOSOME'].apply(lambda x: f"chr{x}" if not re.search('chr',x) else x)
        self.variant_df[['CHROMOSOME','START','END']].drop_duplicates().to_csv(self.vars_chr_bed, sep = '\t', index = False, header = None)
        
        # Limit gnomAD
        self.gnomad_path = os.path.join(self.data_dir,'gnomad_frequencies.bed')
        self.temp_gnomad_path = os.path.join(self.temp_data_dir,'gnomad_frequencies.bed')
        a = pybedtools.BedTool(self.gnomad_path)
        a.intersect(self.vars_chr_bed, u = True).saveas(self.temp_gnomad_path)
        # worker(f"{os.path.join(self.conda_bin, 'bedtools')} intersect -a {self.gnomad_path} -b {self.vars_chr_bed} > {self.temp_gnomad_path}")

        # Limit ClinVar
        self.clinvar_path = os.path.join(self.data_dir, 'clinvar.vcf.gz')
        self.temp_clinvar_path = os.path.join(self.temp_data_dir, 'clinvar.vcf')
        a = pybedtools.BedTool(self.clinvar_path)
        a.intersect(self.vars_bed, header = True, u = True).saveas(self.temp_clinvar_path)
        worker(f"{os.path.join(self.conda_bin, 'bcftools')} view -i \"INFO/CLNSIG[*] == 'Likely_pathogenic' | INFO/CLNSIG[*] == 'Pathogenic'\" -Oz -o {self.temp_clinvar_path}.gz {self.temp_clinvar_path}")
        self.temp_clinvar_path += '.gz'
        worker(f"{os.path.join(self.conda_bin, 'tabix')} -f {self.temp_clinvar_path}")

        # Limit exome data
        self.exome_path = os.path.join(self.data_dir, 'exons.bed')
        self.temp_exome_path = os.path.join(self.temp_data_dir, 'exons.bed')
        a = pybedtools.BedTool(self.exome_path)
        a.intersect(self.vars_chr_bed, u = True).saveas(self.temp_exome_path)
        # worker(f"{os.path.join(self.conda_bin, 'bedtools')} intersect -a {self.exome_path} -b {self.vars_chr_bed} > {self.temp_exome_path}")

        # Limit regulatory regions
        self.reg_path = os.path.join(self.package_data_dir, 'reg_regions.bed')
        self.temp_reg_path = os.path.join(self.temp_data_dir, 'reg_regions.bed')
        a = pybedtools.BedTool(self.reg_path)
        a.intersect(self.vars_chr_bed, u = True).saveas(self.temp_reg_path)
        # worker(f"{os.path.join(self.conda_bin, 'bedtools')} intersect -a {self.reg_path} -b {self.vars_chr_bed} > {self.temp_reg_path}")

        # Get other resources
        self.ref = os.path.join(self.data_dir, 'hg38.fa.gz')
        self.centromere_df = pd.read_csv(os.path.join(self.data_dir, 'centromeres.csv'), index_col = 0)
        self.telomere_df = pd.read_csv(os.path.join(self.data_dir, 'telomeres.csv'),  index_col = 0)
        self.gene_position_df = pd.read_csv(os.path.join(self.package_data_dir, 'gene_positions.csv')).rename(columns = {
            'Chromosomes':'CHROMOSOME',
            'GRCh38_start':'START',
            'GRCh38_stop':'END'
        })
        self.gene_position_df['CHROMOSOME'] = self.gene_position_df['CHROMOSOME'].apply(lambda x: f"chr{x}")
        self.gene_position = {chrom: self.gene_position_df[self.gene_position_df['CHROMOSOME'] == chrom] for chrom in self.gene_position_df['CHROMOSOME'].unique()}
        self.mim2gene = pd.read_csv(os.path.join(self.data_dir, 'mim2gene_medgen'), sep = '\t')
        self.mim2gene.loc[self.mim2gene['GeneID'].str.contains('-'), 'GeneID'] = -999
        self.mim2gene = self.mim2gene.astype({'GeneID': int})
        self.mim2gene.columns = ['OMIM ID'] + self.mim2gene.columns[1:].to_list()

        try:
            self.gnomadSV_df = pd.read_csv(self.temp_gnomad_path, sep = '\t')
            self.gnomadSV_df.columns = ['CHROMOSOME','START','END','TYPE','POP_FREQ']
            self.gnomadSV_df = self.gnomadSV_df.astype({'CHROMOSOME': str})
            # self.gnomadSV_df['CHROMOSOME'] = self.gnomadSV_df['CHROMOSOME'].str[3:]
            self.gnomadSV = {t: {chrom: self.gnomadSV_df[(self.gnomadSV_df['CHROMOSOME'] == chrom) & (self.gnomadSV_df['TYPE'] == t)] for chrom in self.gnomadSV_df['CHROMOSOME'].unique()} for t in ['DEL','DUP']}
        except:
            self.gnomadSV = {}

        try:
            self.exome_df = pd.read_csv(self.temp_exome_path, sep = '\t')
            self.exome_df.columns = ['CHROMOSOME','START','END','SENSE','ID']
            self.exome_df = self.exome_df.astype({'CHROMOSOME': str})
            # self.gnomadSV_df['CHROMOSOME'] = self.gnomadSV_df['CHROMOSOME'].str[3:]
            self.exon_data = {chrom: self.exome_df[(self.exome_df['CHROMOSOME'] == chrom)] for chrom in self.exome_df['CHROMOSOME'].unique()}
        except:
            self.exon_data = {}

        try:
            self.reg_df = pd.read_csv(self.temp_reg_path, sep = '\t')
            self.reg_df.columns = ['CHROMOSOME','START','END','ID']
            self.reg_data = {chrom: self.reg_df[(self.reg_df['CHROMOSOME'] == chrom)] for chrom in self.reg_df['CHROMOSOME'].unique()}
        except:
            self.reg_data = {}

        try:
            self.hi_ts_df = pd.read_csv(os.path.join(self.data_dir, 'hi_ts_parsed.csv'))
            self.hi_ts_df['CHROMOSOME'] = self.hi_ts_df['CHROMOSOME'].apply(lambda x: f"chr{x}")
            self.hi_ts = {chrom: self.hi_ts_df[self.hi_ts_df['CHROMOSOME'] == chrom] for chrom in self.hi_ts_df['CHROMOSOME'].unique()}
        except:
            self.hi_ts = {}

        # Create a breakpoint dataframe to determine chromosome arm length
        self.breakpoint_df = self.centromere_df.merge(self.telomere_df, on = 'CHROMOSOME', suffixes = ('_cen','_tel'))
        self.breakpoint_df['arm1_len'] = self.breakpoint_df['start_cen'] -  self.breakpoint_df['start_tel']
        self.breakpoint_df['arm2_len'] = self.breakpoint_df['end_tel'] - self.breakpoint_df['end_cen']

        # Conservation data
        # cols = ['bin', 'CHROMOSOME', 'minPos', 'maxPos', 'name', 'width', 'count', 'offset', 'wibName', 'lowerLimit', 'dataRange', 'validCount', 'sumData', 'sumSquares']
        # cols_ = ['CHROMOSOME','minPos','maxPos','width','count','offset','lowerLimit','dataRange']
        # col_types = {'minPos':int,'maxPos':int,'width':int,'count':int,'offset':int,'lowerLimit':float,'dataRange':float}

        # PhyloP
        self.phlyop_bw = pyBigWig.open(os.path.join(self.data_dir,'phylop_mvg_avg.bw'))
        # self.phylop = pd.read_csv(os.path.join(self.data_dir,'phyloP100way.txt.gz'), sep = '\t')
        # self.phylop.columns = cols
        # self.phylop = self.phylop[cols_].astype(col_types)
        # self.phylop['CHROMOSOME'] = self.phylop['CHROMOSOME'].str[3:]
        # self.phylop = {chrom: self.phylop[self.phylop['CHROMOSOME'] == chrom] for chrom in self.phylop['CHROMOSOME'].unique()}
        # with open(os.path.join(self.data_dir,'phyloP100way.wib'),'rb') as f:
        #     self.phylop_wib = f.read()

        # phastCons
        self.phastcons_bw = pyBigWig.open(os.path.join(self.data_dir,'phastcons_mvg_avg.bw'))
        # self.phastcons = pd.read_csv(os.path.join(self.data_dir,'phastCons100way.txt.gz'), sep = '\t')
        # self.phastcons.columns = cols
        # self.phastcons = self.phastcons[cols_].astype(col_types)
        # self.phastcons['CHROMOSOME'] = self.phastcons['CHROMOSOME'].str[3:]
        # self.phastcons = {chrom: self.phastcons[self.phastcons['CHROMOSOME'] == chrom] for chrom in self.phastcons['CHROMOSOME'].unique()}
        # with open(os.path.join(self.data_dir,'phastCons100way.wib'),'rb') as f:
        #     self.phastcons_wib = f.read()

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


    def remove_temp_files(self):
        """
        Remove temporary files created during the feature building process
        """
        shutil.rmtree(self.temp_data_dir)
        

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

            if row['CHROMOSOME'] not in self.gene_position.keys():
                return []

            gene_position = self.gene_position[row['CHROMOSOME']]
            df = gene_position.loc[
                        (gene_position['START'].between(row['START'], row['END']))
                            |
                        (gene_position['END'].between(row['START'], row['END']))
                            |
                        ((gene_position['START'] <= row['START']) & (gene_position['END'] >= row['END']))
                ]    

            if df.empty:
                return []
                
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
            cen_start, cen_end = self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'start_cen'], self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'end_cen']
            arm1_len, arm2_len = self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'arm1_len'], self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'arm2_len']
            if row['END'] < cen_start:
                # return (cen_start - row['END']) / arm1_len
                return cen_start - row['END']
            else:
                # return (row['START'] - cen_end) / arm2_len
                return row['START'] - cen_end


        def get_telomere_distance(row):
            """
            Use the previously downloaded telomere coordinate data to obtain the 
            distance of the CNV to the telomere (normalized by chromosome arm 
            length)
            """
            tel_start, tel_end = self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'start_tel'], self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'end_tel']
            cen_start, cen_end = self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'start_cen'], self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'end_cen']
            arm1_len, arm2_len = self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'arm1_len'], self.breakpoint_df.loc[f"{row['CHROMOSOME']}", 'arm2_len']
            if row['END'] < cen_start:
                # return (row['START'] - tel_start) / arm1_len
                return row['START'] - tel_start
            else:
                # return (tel_end - row['END']) / arm2_len
                return tel_end - row['END']


        def get_gc_content(row):
            command = f"{os.path.join(self.conda_bin,'samtools')} faidx {self.ref} {row['CHROMOSOME']}:{row['START']}-{row['END']}"
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

            if self.hi_ts == {}:
                return self.null_hi_ts

            if row['CHROMOSOME'] not in self.hi_ts.keys():
                return self.null_hi_ts

            hi_ts_df = self.hi_ts[row['CHROMOSOME']]

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

            if self.gnomadSV == {}:
                return 0

            if row['CHANGE'] not in self.gnomadSV.keys():
                return 0

            if row['CHROMOSOME'] not in self.gnomadSV[row['CHANGE']].keys():
                return 0

            gnomad_df = self.gnomadSV[row['CHANGE']][row['CHROMOSOME']]

            df = gnomad_df.loc[
                (gnomad_df['START'].between(row['START'], row['END']))
                    |
                (gnomad_df['END'].between(row['START'], row['END']))
                    |
                ((gnomad_df['START'] <= row['START']) & (gnomad_df['END'] >= row['END']))
            ]

            if df.empty:
                return 0
            
            df['percent_coverage'] = (
            df[['START', 'END']].clip(lower=row['START'], upper=row['END']).diff(axis=1) / 
                        df[['START', 'END']].diff(axis=1)
                        )['END']

            # Only consider variants that cover at least half of the CNV
            df = df.loc[df['percent_coverage'] >= 0.5]

            # return gnomad_df['pop_freq'].mean()
            return df['POP_FREQ'].max()

        
        def get_clinvar_path_density(row, clinvar_reader):
            try:
                query = clinvar_reader.fetch(chrom = row['CHROMOSOME'][3:], start = row['START'], end = row['END'])
                return len([q for q in query])
            except:
                return 0
            

        # def moving_average(values, window_size):
        #     """
        #     Calculate the moving average using a specified window size.

        #     Parameters:
        #     values (array-like): The input data for which the moving average is computed.
        #     window_size (int): The size of the moving window.

        #     Returns:
        #     array-like: The moving average of the input data.
        #     """
        #     if window_size < 1:
        #         raise ValueError("Window size must be at least 1")

        #     # Create a window: using a uniform weight (simple moving average)
        #     window = np.ones(int(window_size)) / float(window_size)

        #     # Compute the moving average using convolution
        #     moving_avg = np.convolve(values, window, 'valid')

        #     return moving_avg


        # def get_wib_values(cons_df, wib_data):

        #     chroms, starts, steps, values = [], [], [], []

        #     for idx, row in cons_df.iterrows():

        #         for i in range(row['count']):
        #             byte_value = wib_data[row['offset'] + i]
        #             if byte_value < 128:
        #                 value = row['lowerLimit'] + (row['dataRange'] * (byte_value / 127.0))
        #                 chroms.append(row['CHROMOSOME'])
        #                 starts.append(row['minPos'] + i * row['width'])
        #                 steps.append(row['width'])
        #                 values.append(value)

        #     return pd.DataFrame({
        #         'chrom': chroms,
        #         'start': starts,
        #         'step': steps,
        #         'value': values
        #     })


        # def get_cons(row, window_size, cons, wib):
        def get_cons(row, window_size, bw):

            # # Get necessary rows from wiggle
            # cons_df = cons[row['CHROMOSOME']]
            # cons_df = cons_df[
            #     (row['START'] - window_size <= cons_df['maxPos'] )
            #         &
            #     (cons_df['minPos'] <= row['END'] + window_size)
            # ]

            # # Get wib values
            # wib_values = get_wib_values(cons_df, wib)
            # wib_values = wib_values[['start','value']].rename(columns = 
            #                                         {'start':'POS','value':'VALUE'})
            # wib_values = wib_values[wib_values['POS'].between(
            #     row['START'] - (window_size / 2), row['END'] + (window_size / 2))]
            try: # Take maximum
                return bw.stats(f"{row['CHROMOSOME']}", int(row['START']), int(row['END']), type = 'max')[0]

            except:

                try:
                    return bw.stats(f"{row['CHROMOSOME']}", int(row['START']), int(row['END']) + 1, type = 'max')[0]


                except Exception as e:
                    print(f"{row['CHROMOSOME']}", row['START'], row['END'])
                    print(type(e), e)
                    return 0


        def get_exon_count(row):
            """
            Get the number of exons that are spanned by the CNV
            """
            if self.exon_data == {}:
                return 0

            if row['CHROMOSOME'] not in self.exon_data.keys():
                return 0    

            exon_df = self.exon_data[row['CHROMOSOME']]
            df = exon_df.loc[
                        (exon_df['START'].between(row['START'], row['END']))
                            |
                        (exon_df['END'].between(row['START'], row['END']))
                            |
                        ((exon_df['START'] <= row['START']) & (exon_df['END'] >= row['END']))
                ]

            return len(df.index)


        def get_reg_count(row):
            """
            Get the number of regulatory regions that are spanned by the CNV
            """

            if self.reg_data == {}:
                return 0

            if row['CHROMOSOME'] not in self.reg_data.keys():
                return 0    

            reg_df = self.reg_data[row['CHROMOSOME']]
            df = reg_df.loc[
                        (reg_df['START'].between(row['START'], row['END']))
                            |
                        (reg_df['END'].between(row['START'], row['END']))
                            |
                        ((reg_df['START'] <= row['START']) & (reg_df['END'] >= row['END']))
                ]

            return len(df.index)

        cnv_df = self.variant_df.copy()

        # Intialize progress bar
        bar_widgets = [progressbar.FormatLabel('Building Features'), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.Timer()]
        bar = progressbar.ProgressBar(maxval=15, \
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

        bar_widgets[0] = progressbar.FormatLabel('Calculating genomic position')
        bar.update(5)
        cnv_df['CENT_DIST'] = cnv_df.apply(get_centromere_distance, axis = 1)
        cnv_df['TEL_DIST'] = cnv_df.apply(get_telomere_distance, axis = 1)

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
        try:
            clinvar_reader = vcf.Reader(filename = self.temp_clinvar_path, compressed=True, encoding='ISO-8859-1')
        except:
            clinvar_reader = None
        cnv_df['CLINVAR_DENSITY'] = cnv_df.apply(get_clinvar_path_density, clinvar_reader = clinvar_reader, axis = 1)
        
        bar_widgets[0] = progressbar.FormatLabel('Calculating conservation scores')
        bar.update(10)
        cnv_df['PHYLOP_SCORE'] = cnv_df.apply(
            get_cons, 
            window_size = 200,
            # cons = self.phylop,
            # wib = self.phylop_wib,
            bw = self.phlyop_bw,
            axis = 1)
        cnv_df['PHASTCONS_SCORE'] = cnv_df.apply(
            get_cons, 
            window_size = 5000,
            # cons = self.phastcons,
            # wib = self.phastcons_wib,
            bw = self.phastcons_bw,
            axis = 1)
        bar_widgets[0] = progressbar.FormatLabel('Calculating exon count')
        bar.update(12)

        cnv_df['EXON_COUNT'] = cnv_df.apply(get_exon_count, axis = 1)
        bar_widgets[0] = progressbar.FormatLabel('Calculating regulatory region count')
        bar.update(13)

        cnv_df['REG_COUNT'] = cnv_df.apply(get_reg_count, axis = 1)
        bar_widgets[0] = progressbar.FormatLabel('All features generated, normalizing')
        bar.update(14)

        # Fill NAs
        na_vals = {
            'GC_CONTENT': cnv_df['GC_CONTENT'].median(),
            'CLINVAR_DENSITY': 0,
            'POP_FREQ': 0,
            'PHYLOP_SCORE': cnv_df['PHYLOP_SCORE'].min(),
            'PHASTCONS_SCORE': cnv_df['PHASTCONS_SCORE'].min()
        }
        cnv_df.astype({x:float for x in na_vals.keys()}).astype({'CLINVAR_DENSITY':int})
        cnv_df = cnv_df.fillna(na_vals)

        cols_to_normalize = [
            'BP_LEN','GENE_COUNT','DISEASE_COUNT','EXON_COUNT','REG_COUNT',
            'CENT_DIST','TEL_DIST','%HI','pLI', 'LOEUF','GC_CONTENT','POP_FREQ',
            'CLINVAR_DENSITY','PHYLOP_SCORE','PHASTCONS_SCORE']

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
        bar.update(15)
        bar.finish()

        cols = ['CHROMOSOME','START','END','CHANGE'] 
        cols = cols + [oc for oc in original_cols if oc not in cols] + ['gene_info'] + cols_to_normalize + hs_ts_cols
        cols = [c for c in cols if c not in ['build','HI_SCORE','TS_SCORE']]

        self.feature_df = cnv_df[cols]
        
