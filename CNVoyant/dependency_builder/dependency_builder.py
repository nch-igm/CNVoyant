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

    def __init__(self, data_dir: str, gnomadSV_version: str = '2.1'):

        self.conda_bin = os.path.join(sys.exec_prefix, 'bin')
        self.data_dir = data_dir
        self.package_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.gnomadSV_version = gnomadSV_version
        self.repo_url = 'https://github.com/nch-igm/CNVoyant'


    def worker(self, cmd):
        parsed_cmd = shlex.split(cmd)
        p = subprocess.Popen(parsed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        out, err = p.communicate()
        return out.decode() if out else err.decode()


    def download_file(self, url, output, binary=False):
        """
        Download a file from `url` and save it locally under `output`.
        The file is downloaded in chunks to save on memory usage.
        """
        r = requests.get(url)
        if r.status_code == 200:
            if binary:
                with open(output, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                with open(output, 'w') as f:
                    f.write(r.text)
            print(f"File downloaded successfully: {output}")
        else:
            print(f'Bad url: {url}')


    def download_lfs_files(self):
        """
        Download a necessary file from the CNVoyant Git LFS-enabled repository.
        """
        for variant_type in ['del','dup']:
            
            file_path = os.path.join('CNVoyant','classifier','models',f'{variant_type}_model.pickle')
            output = os.path.join(self.data_dir, os.path.basename(file_path))
            url = f"{os.path.join(self.repo_url,'blob','main',file_path)}?raw=true"
            self.download_file(url, output)



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
        # df.columns = ['chrom','start','stop','type','pop_freq']
        df.to_csv(output_path, index = False, sep = '\t')

        
    
    def build_gnomad_db(self):
        """
        gnomadSV data is needed in order to generate population frequencies for 
        the indicated CNVs. It is larger when unzipped (~31MB). 
        """
        gnomad_path = os.path.join(self.package_data_dir,'gnomad4.bed.gz')
        gnomad_df_path = os.path.join(self.data_dir, 'gnomad_frequencies.bed')
        if not os.path.exists(gnomad_df_path):
            
            df = pd.read_csv(gnomad_path, sep = '\t')
            df.to_csv(gnomad_df_path, sep = '\t', index = False, header = None)

            # Download gnomAD
            # url = f"https://storage.googleapis.com/gcp-public-data--gnomad/papers/2019-sv/gnomad_v{self.gnomadSV_version}_sv.sites.vcf.gz"
            # self.download_file(url, gnomad_path, binary = True)

            # # Parse gnomAD
            # self.get_gnomad_frequencies(gnomad_path, gnomad_df_path)


    def get_reference(self):
        """
        hg38 reference genome is required (~1GB).
        """
        reference_path = os.path.join(self.data_dir, 'hg38.fa.gz')
        reference_unzipped_path = os.path.join(self.data_dir, 'hg38.fa')
        if not os.path.exists(reference_path):

            # Download reference
            url = 'https://hgdownload2.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
            self.download_file(url, reference_path, binary = True)

            # Reformat reference
            self.worker(f"{os.path.join(self.conda_bin,'gunzip')} {reference_path}")
            self.worker(f"{os.path.join(self.conda_bin,'bgzip')} -f {reference_unzipped_path}")


    def get_cons_scores(self):
        """
        phyloP and phastCons scores are required to generate estimations of 
        region conservation. The databases containg these scores weigh in at
        around (~5GB).
        """

        conservation_links = [
        #     'https://hgdownload.soe.ucsc.edu/gbdb/hg38/multiz100way/phastCons100way.wib',
        #     'https://hgdownload.soe.ucsc.edu/gbdb/hg38/multiz100way/phyloP100way.wib',
        #     'https://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/phastCons100way.txt.gz',
        #     'https://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/phyloP100way.txt.gz',
              'https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw',
              'https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw'
        ]

        for cl in conservation_links:
            cons_path = os.path.join(self.data_dir, os.path.basename(cl))
            if not os.path.exists(cons_path):
                self.download_file(cl, cons_path, binary = True)


    def build_breakpoints(self):
        """
        To calculate distances to centromeres and telomeres, a data frame
        containing this data must be generated. If the cytoband data for hg38 is
        not already available, it will be downloaded into the folder indicated 
        in the config file (cnv_data).
        """
        bp_path = os.path.join(self.data_dir, 'hg38_cytoband.tsv.gz')
        if not os.path.exists(bp_path):
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz"
            self.download_file(url, bp_path, binary = True)

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

        hi_ts_df_path = os.path.join(self.package_data_dir, 'hi_ts_regions.csv')
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


    def build_clinvar_db(self):
        """
        Download the hg38 ClinVar short variant database to find short pathogenic
        variant density
        """
        clinvar_short_path = os.path.join(self.data_dir, 'clinvar.vcf.gz')
        if not os.path.exists(clinvar_short_path):
            clinvar_url = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz'
            self.download_file(clinvar_url, clinvar_short_path, binary = True)
            

        if not os.path.exists(f"{clinvar_short_path}.tbi"):

            # Create an index file
            cmd = f"""
            {os.path.join(self.conda_bin, 'tabix')} -f {clinvar_short_path}
            """
            p = self.worker(cmd)


    def get_mim2gene(self):
        """
        Get the mim2gene data
        """
        mim2_gene_url = 'https://ftp.ncbi.nih.gov/gene/DATA/mim2gene_medgen'
        output_file = os.path.join(self.data_dir, 'mim2gene_medgen')
        if not os.path.exists(output_file):
            self.download_file(mim2_gene_url, output_file)


    def build_all(self):
        """
        Build all dependencies and update a progress bar as we chug along
        """

        # Intialize progress bar
        bar_widgets = [progressbar.FormatLabel('Intializing'), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.Timer()]
        bar = progressbar.ProgressBar(maxval=8, \
            widgets=bar_widgets)
        bar.start()

        # Download and unpack gnomAD data
        # self.build_gnomad_db()
        bar_widgets[0] = progressbar.FormatLabel('Downloading chromosomal breakpoints')
        bar.update(1)
        
        # Get chromosomal breakpoint data
        self.build_breakpoints()
        bar_widgets[0] = progressbar.FormatLabel('Downloading HI/TS data')
        bar.update(2)

        # Download and unpack haploinsufficiency (HI) and triplosensitivity (TS) data
        self.build_hi_ts_data()
        bar_widgets[0] = progressbar.FormatLabel('Downloading ClinVar short variants')
        bar.update(3)

        # Download and unpack ClinVar short variants
        self.build_clinvar_db()
        bar_widgets[0] = progressbar.FormatLabel('Downloading mim2gene file')
        bar.update(4)

        # Download OMIM to gene annotations
        self.get_mim2gene()
        bar_widgets[0] = progressbar.FormatLabel(f'Unpacking gnomAD (v4)')
        bar.update(5)

        # Download gnomAD
        self.build_gnomad_db()
        bar_widgets[0] = progressbar.FormatLabel('Downloading reference (GRCh38)')
        bar.update(6)

        # Download reference
        self.get_reference()
        bar_widgets[0] = progressbar.FormatLabel('Downloading conservation scores')
        bar.update(7)

        # Download conservation scores
        self.get_cons_scores()
        bar_widgets[0] = progressbar.FormatLabel('Dependencies ready')
        bar.update(8)

        bar.finish()
