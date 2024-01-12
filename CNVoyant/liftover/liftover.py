import os
import uuid
import shlex
import subprocess
import json
import pandas as pd


def worker(cmd):
    parsed_cmd = shlex.split(cmd)
    p = subprocess.Popen(parsed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    out, err = p.communicate()
    return out.decode() if out else err.decode()


def get_liftover_data():
    with open(os.path.join(os.path.dirname(__file__), 'paths.json'),'r') as f:
        liftover_data = json.load(f)
    liftover_data = {k:v if v.startswith('/') else os.path.join(os.path.dirname(__file__), v) for k,v in liftover_data.items()}
    return liftover_data


def get_liftover_positions(row):
    """
    Get the GRCh38 positions for every CNV. Requires a pointer to a
    previously downloaded liftover executable. Can be downloaded with the 
    DependencyBuilder object.
    """

    data = get_liftover_data()
    
    coordinates = {}
    liftover_executable_path = data['liftover_executable_path']
    input_filename = os.path.join(data['liftover_temp_dir'], str(uuid.uuid4()))
    lifted_output_filename = os.path.join(data['liftover_temp_dir'], str(uuid.uuid4()))
    unlifted_output_filename = os.path.join(data['liftover_temp_dir'], str(uuid.uuid4()))
            
    if row['build'] == 'GRCh38':
        return {'START': row['START'], 'END': row['END']}

    # Write coordinates to a temporary file
    pd.DataFrame({
        'chrom': ['chr' + str(row['CHROMOSOME'])],
        'pos': [row['START']],
        'pos1': [row['END']]
    }).to_csv(input_filename, sep = '\t', index = False, header = False)

    # Pass to the liftover command line executable
    chain_file = data['liftover_hg19tohg38_chain_path']
    cmd = f'{liftover_executable_path} {input_filename} {chain_file} {lifted_output_filename} {unlifted_output_filename}'
    p = worker(cmd)

    # Read in the results
    try:
        lifted_coordinates = pd.read_csv(lifted_output_filename, sep = '\t', header = None)
    except:
        print(f"Failed to liftover at {'chr' + row['CHROMOSOME']}:{row['START']}-{row['END']}")
        lifted_coordinates = pd.read_csv(unlifted_output_filename, sep = '\t', header = None, comment='#')
    coordinates['START'] = lifted_coordinates.iloc[0, 1]
    coordinates['END'] = lifted_coordinates.iloc[0, 2]
    
    # Remove temp files
    for f in [input_filename, lifted_output_filename, unlifted_output_filename]:
        os.remove(f)
    
    return coordinates
