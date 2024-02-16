import os
import sys
import subprocess
import shlex
import pandas as pd
import requests
import pandas as pd
import numpy as np
import os
import re
import time
import traceback
import progressbar
import vcf
from ..liftover.liftover import get_liftover_positions


class DependencyBuilder:

    def __init__(self, data_dir: str = os.path.join(os.path.dirname(__file__), '..', 'data'), gnomadSV_version: str = '2.1'):
        self.data_dir = data_dir
        self.gnomadSV_version = gnomadSV_version
        self.repo_url = 'https://github.com/yourusername/yourrepository.git'
        self.target_dir = './path/to/save/lfs/files'


    def worker(self, cmd):
        parsed_cmd = shlex.split(cmd)
        p = subprocess.Popen(parsed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        out, err = p.communicate()
        return out.decode() if out else err.decode()


    def _is_git_lfs_installed(self):
        try:
            subprocess.run(['git', 'lfs', 'version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False


    def get_lfs_files(self):

        # Check if Git LFS is installed
        if not self._is_git_lfs_installed():
            print("Git LFS is not installed. Please install Git LFS first.")
            return False
        
        # Clone LFS files or pull from the repository
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
        p = worker(f"git lfs clone {self.repo_url} {self.target_dir}")
        
        # Additional steps could include verifying the integrity of the files
        print("LFS files downloaded successfully.")
        return True


    def get_gnomad_frequencies(self, vcf_path, output_path):

        vcf_reader = vcf.Reader(filename = vcf_path, compressed=True, encoding='ISO-8859-1')

        rows = []

        for record in vcf_reader:
            # print(record.INFO['SVTYPE'], record.FILTER)
            if record.INFO['SVTYPE'] in ('DEL','DUP') and record.FILTER == []:
                chrom = record.CHROM
                pos1 = record.POS
                pos2 = record.INFO['END']
                var_type = 'DUP' if record.INFO['SVTYPE'] == 'DUP' else 'DEL'
                freq = record.INFO['AC'][0] / record.INFO['AN']
                rows.append([chrom, pos1, pos2, var_type, freq])

        df = pd.DataFrame(rows)
        df.columns = ['chrom','start','stop','type','pop_freq']
        df.to_csv(output_path, index = False)

        
    
    def build_gnomad_db(self):
        """
        gnomadSV data is needed in order to generate population frequencies for 
        the indicated CNVs. It is quite large when unzipped (~2GB). Be sure to 
        account for this before running.
        """
        gnomad_path = os.path.join(self.data_dir, f'gnomad_v{self.gnomadSV_version}_sv.sites.vcf.gz')
        gnomad_df_path = os.path.join(self.data_dir, 'gnomad_frequencies.csv')
        if not os.path.exists(gnomad_df_path):
            url = f"https://storage.googleapis.com/gcp-public-data--gnomad/papers/2019-sv/gnomad_v{self.gnomadSV_version}_sv.sites.vcf.gz"
            cmd = f"""
                curl -s "{url}" -o {gnomad_path}
            """
            p = self.worker(cmd)

            # Parse gnomAD
            self.get_gnomad_frequencies(gnomad_path, gnomad_df_path)


    def build_breakpoints(self):
        """
        To calculate distances to centromeres and telomeres, a data frame
        containing this data must be generated. If the cytoband data for hg38 is
        not already available, it will be downloaded into the folder indicated 
        in the config file (cnv_data).
        """
        bp_path = os.path.join(self.data_dir, 'hg38_cytoband.tsv')
        if not os.path.exists(bp_path):
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz"
            cmd = f"""
                curl -s "{url}" | gunzip -c > {bp_path}
            """
            p = subprocess.Popen(cmd,  stdout=subprocess.PIPE, shell = True)
            out, err = p.communicate()
            if err is not None:
                error_message = f"""
                Failed to download the cytogeneic coordinate data at:
                {url}
                """
                raise error_message

        centromere_df_path = os.path.join(self.data_dir, 'centromeres.csv')
        if not os.path.exists(centromere_df_path):

            # Build centromere data
            centromere_df = pd.read_csv(bp_path, sep = '\t', header = None)
            centromere_df.columns = ['CHROMOSOME', 'start', 'end', 'cytoband', 'info']
            centromere_df = centromere_df.loc[centromere_df['info'] == 'acen']
            centromere_df = centromere_df.iloc[:, 0:3]
            centromere_df = centromere_df.groupby('CHROMOSOME').agg({'start': ['min'], 'end': ['max']})
            centromere_df.columns = ['start', 'end']
            centromere_df.to_csv(centromere_df_path)

        telomere_df_path = os.path.join(self.data_dir, 'telomeres.csv')
        if not os.path.exists(telomere_df_path):

            # Build telomere data
            telomere_df = pd.read_csv(bp_path, sep = '\t', header = None).iloc[:, 0:3]
            telomere_df.columns = ['CHROMOSOME', 'start', 'end']
            telomere_df = telomere_df.groupby('CHROMOSOME').agg({'start': ['min'], 'end': ['max']})
            telomere_df.columns = ['start', 'end']
            telomere_df = pd.DataFrame(index = centromere_df.index).merge(telomere_df, how = 'left', left_index = True, right_index = True)
            telomere_df.to_csv(telomere_df_path)


    def build_hi_ts_data(self):
        """
        Data downloaded from https://search.clinicalgenome.org/kb/gene-dosage. 
        This data comes packaged with CNVoyant, this function parses this data 
        for use in feature generation.
        """

        hi_ts_df_path = os.path.join(self.data_dir, 'hi_ti_gene.csv')
        hi_ts_parsed_path = os.path.join(self.data_dir, 'hi_ts_parsed.csv')
        if not os.path.exists(hi_ts_parsed_path):

            def parse_coordinates(row):
                pos = str(row['GRCh38'])
                pos_data = {}
                pos_data['CHROMOSOME'] = pos[:pos.find(':')]
                pos_data['START'] = pos[pos.find(':') + 1:pos.find('-')]
                pos_data['END'] = pos[pos.find('-') + 1:]
                return pos_data
            
            hi_ts_df = pd.read_csv(hi_ts_df_path, index_col=0).reset_index().rename(columns = {'index': 'type'})
            hi_ts_df['build'] = 'GRCh38'
            hi_ts_df['pos_data'] = hi_ts_df.apply(parse_coordinates, axis = 1)
            hi_ts_df = pd.concat([hi_ts_df['pos_data'].apply(pd.Series), hi_ts_df.drop(['pos_data','GRCh38'], axis = 1)], axis = 1)
            hi_ts_df = hi_ts_df.loc[~hi_ts_df['CHROMOSOME'].isin(['', 'na'])]
            hi_ts_df = hi_ts_df.astype({'START': int, 'END': int})
            
            hi_ts_df['pos_data'] = hi_ts_df.apply(get_liftover_positions, axis = 1)
            hi_ts_df = pd.concat([hi_ts_df.drop(['pos_data','START','END'], axis = 1), hi_ts_df['pos_data'].apply(pd.Series)], axis = 1)
            
            # Update TS/HI score when unknown
            cols = ['HI Score','TS Score']
            for c in cols:
                # hi_ts_df[c] = hi_ts_df[c].replace('Not Yet Evaluated','0 (No Evidence)')
                hi_ts_df[c.upper().replace(' ','_')] = hi_ts_df[c]
            hi_ts_df = hi_ts_df.drop(columns = cols)

            # Only keep genes
            hi_ts_df.loc[hi_ts_df['%HI'] == '‐', '%HI'] = pd.to_numeric(hi_ts_df['%HI'], errors = 'coerce').max()
            hi_ts_df.loc[hi_ts_df['pLI'] == '‐', 'pLI'] = pd.to_numeric(hi_ts_df['pLI'], errors = 'coerce').max()
            hi_ts_df.loc[hi_ts_df['LOEUF'] == '‐', 'LOEUF'] = pd.to_numeric(hi_ts_df['LOEUF'], errors = 'coerce').min()

            # Break up
            hi_ts = ['HI','TS']
            hi_ts_map = {
                'Not Yet Evaluated':'NOT_EVALUATED',
                '0 (No Evidence)':'NO_EVIDENCE',
                '1 (Little Evidence)':'LITTLE_EVIDENCE',
                '2 (Emerging Evidence)':'EMERGING_EVIDENCE',
                '3 (Sufficient Evidence)':'SUFFICIENT_EVIDENCE',
                '30 (Autosomal Recessive)':'AUTOSOMAL_RECESSIVE',
                '40 (Dosage Sensitivity Unlikely)':'UNLIKELY'
            }

            null_cols = {}
            for ht in hi_ts:
                hi_ts_df[f'{ht}_SCORE'] = hi_ts_df[f'{ht}_SCORE'].replace(hi_ts_map)
                for v in hi_ts_map.values():
                    hi_ts_df.loc[hi_ts_df[f'{ht}_SCORE'] == v, f'{ht}_{v}'] = 1
                    null_cols[f'{ht}_{v}'] = 0

            hi_ts_df = hi_ts_df.fillna(value=null_cols)
            hi_ts_df.to_csv(hi_ts_parsed_path, index = False)

        # Build scoring map
        #     hi_ts_mapped_df.to_csv(hi_ts_map_path, index = False)


    def build_clinvar_db(self):
        """
        Download the hg38 ClinVar short variant database to find short pathogenic
        variant density
        """
        clinvar_short_path = os.path.join(self.data_dir, 'clinvar.vcf.gz')
        if not os.path.exists(clinvar_short_path):
            clinvar_url = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz'
            # Download the vcf
            cmd = f"""
            curl -s "{clinvar_url}" -o {clinvar_short_path} 
            """
            p = self.worker(cmd)

        if not os.path.exists(f"{clinvar_short_path}.tbi"):

            # Create an index file
            cmd = f"""
            tabix -f {clinvar_short_path}
            """
            p = self.worker(cmd)


    def get_mim2gene(self):
        """
        Get the mim2gene data
        """
        mim2_gene_url = 'https://ftp.ncbi.nih.gov/gene/DATA/mim2gene_medgen'
        output_file = os.path.join(self.data_dir, 'mim2gene_medgen')
        cmd = f"""
                curl -s "{mim2_gene_url}" -o {output_file}
            """
        p = self.worker(cmd)


    def build_all(self):
        """
        Build all dependencies and update a progress bar as we chug along
        """

        # Intialize progress bar
        bar_widgets = [progressbar.FormatLabel('Intializing'), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.Timer()]
        bar = progressbar.ProgressBar(maxval=5, \
            widgets=bar_widgets)
        bar.start()

        # Download and unpack gnomAD data
        # self.build_gnomad_db()
        bar.update(1)
        bar_widgets[0] = progressbar.FormatLabel('Downloading chromosomal breakpoints')
        
        # Get chromosomal breakpoint data
        self.build_breakpoints()
        bar.update(2)
        bar_widgets[0] = progressbar.FormatLabel('Downloading HI/TS data')

        # Download and unpack haploinsufficiency (HI) and triplosensitivity (TS) data
        self.build_hi_ts_data()
        bar.update(3)
        bar_widgets[0] = progressbar.FormatLabel('Downloading ClinVar short variants')

        # Download and unpack ClinVar short variants
        self.build_clinvar_db()
        bar.update(4)
        bar_widgets[0] = progressbar.FormatLabel('Downloading mim2gene file')

        # Download OMIM to gene annotations
        self.get_mim2gene()
        bar.update(5)
        bar_widgets[0] = progressbar.FormatLabel('Dependencies ready')

        bar.finish()
