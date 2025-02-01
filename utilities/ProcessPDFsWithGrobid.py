# Process PDF files with Grobid
#
# Using:
#      podman pull docker.io/grobid/grobid:0.8.1 
# on the arkinlab server 
# 
# This script processes PDF files with Grobid and saves the results to a directory of JSON files.
#
# Usage:
#   python ProcessPDFsWithGrobid.py <input_directory> <output_directory>
#
# Parameters:
#   input_directory: The directory containing the PDF files to process.
#   output_directory: The directory to save the processed results.
#
# Returns:
#   A directory of JSON files with the processed results.
# 
# Note I have installed grobid server on Arkin Lab servers and started it with the following command:
#    podman run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.1 &
# I use ssh tunneling to access it from my local machine.
#     ssh -L 8070:localhost:8070 aparkin@niya.qb3.berkeley.edu
#
# I have been running this using:
#     python ProcessPDFsWithGrobid.py ../../../ENIGMA\ Project/data/2024\ ENIGMA\ Publications/All\ Papers ../data/grobid_output --batch_size 10
#
# And with config.json set to:'
#{
#    "grobid_server": "http://localhost:8070",
#    "batch_size": 100,
#    "sleep_time": 20,
#    "timeout": 200,
#    "coordinates": [ "persName", "figure", "ref", "biblStruct", "formula", "s", "p", "note", "title", "affliation", "header" ]
#}


import os
import shutil
import argparse
import json
import time
from tqdm import tqdm
from grobid_client.grobid_client import GrobidClient
from lxml import etree

def process_pdfs_with_grobid(input_directory, output_directory, config_path, batch_size):
    # Convert paths to absolute paths
    input_directory = os.path.abspath(input_directory)
    output_directory = os.path.abspath(output_directory)

    # Load configuration from file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    grobid_url = config.get('grobid_server', 'http://localhost:8070')
    batch_size = batch_size if batch_size is not None else config.get('batch_size', 10)
    timeout = config.get('timeout', 60)
    sleep_time = config.get('sleep_time', 5)
    print(f"Using GROBID server at: {grobid_url} with batch size: {batch_size}")  # Debugging line

    client = GrobidClient(config_path=config_path)

    # Test server connection
    try:
        client._test_server_connection()
        print("GROBID server is up and running.")
    except Exception as e:
        print(f"Failed to connect to GROBID server: {e}")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Collect all PDF files
    pdf_files = [os.path.join(root, file)
                 for root, _, files in os.walk(input_directory)
                 for file in files if file.endswith(".pdf")]

    # Process PDFs in batches
    for i in range(0, len(pdf_files), batch_size):
        batch_files = pdf_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} with {len(batch_files)} files.")

        # Create a temporary directory for the current batch
        temp_directory = os.path.join("temp_batch", f"batch_{i // batch_size + 1}")
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)

        # Copy batch files to the temporary directory
        for file in batch_files:
            shutil.copy(file, temp_directory)

        # Measure the time taken for processing
        start_time = time.time()
        success = False
        retries = 5
        while not success and retries > 0:
            try:
                client.process(
                    service='processFulltextDocument',
                    input_path=temp_directory,
                    output=output_directory,
                    consolidate_header=True,
                    consolidate_citations=True,
                    include_raw_affiliations=False,
                    include_raw_citations=False,
                    segment_sentences=True,
                    tei_coordinates=False,
                    force=True,
                    n=20
                )
                success = True
                end_time = time.time()
                print(f"Processed batch {i // batch_size + 1} successfully.")
                print(f"Time taken: {end_time - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
                retries -= 1
                if retries > 0:
                    print(f"Retrying... ({3 - retries} attempts left)")
                    time.sleep(sleep_time)

        # Clean up the temporary directory
        shutil.rmtree(temp_directory)

    # List contents of the output directory
    if False:
        print("\nOutput directory contents:")
        for root, dirs, files in os.walk(output_directory):
            for file in files:
                print(os.path.join(root, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs with GROBID")
    parser.add_argument("input_directory", help="The directory containing the PDF files to process")
    parser.add_argument("output_directory", help="The directory to save the processed results")
    parser.add_argument("--config_path", default="config.json", help="Path to the configuration file (default: config.json)")
    parser.add_argument("--batch_size", type=int, help="Number of files to process in each batch (overrides config.json)")

    args = parser.parse_args()

    process_pdfs_with_grobid(args.input_directory, args.output_directory, args.config_path, args.batch_size)