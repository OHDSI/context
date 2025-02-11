import pathlib
import shutil
import tempfile
import zipfile
from urllib.parse import quote

import polars as pl
import duckdb

output_folder = "/Users/xxx/Desktop"

def process_data_object(data_object_dir):
    zip_path = pathlib.Path(data_object_dir) / "covariates"
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        sql_files = list(tmpdir.glob("*.sqlite")) + list(tmpdir.glob("*.duckdb"))
        if not sql_files:
            raise FileNotFoundError("No supported SQL database file (.sqlite or .duckdb) found in the zip archive.")

        data = str(sql_files[0])
        print(f"Found database file: {data}")

        if pathlib.Path(data).suffix == ".sqlite":
            data = quote(data)
            data_ref = pl.read_database_uri(
                "SELECT * FROM covariateRef", uri=f"sqlite://{data}"
            ).lazy()
        elif pathlib.Path(data).suffix == ".duckdb":
            data = quote(data)
            # Create a local copy for DuckDB.
            destination = pathlib.Path(data).parent.joinpath("python_copy.duckdb")
            path_copy = pathlib.Path(shutil.copy(data, destination))
            conn = duckdb.connect(str(path_copy))
            data_ref = conn.sql("SELECT * FROM covariateRef").pl().lazy()
            conn.close()
            path_copy.unlink()  # Remove the temporary copy.
        else:
            raise ValueError("Only .sqlite and .duckdb files are supported")

        return data_ref

def main():
    data_object_dirs = [
        "/Users/xxx/data/plp/yyy",
        "/Users/xxx/data/plp/yyy",
        "/Users/xxx/data/plp/yyy"
    ]

    all_concept_ids = set()

    for dir_path in data_object_dirs:
        print(f"\nProcessing data object in {dir_path} ...")
        try:
            data_ref = process_data_object(dir_path)
            df = data_ref.collect()
            concept_ids = df["conceptId"].to_list()
            for cid in concept_ids:
                try:
                    all_concept_ids.add(int(cid))
                except Exception as ex:
                    print(f"Could not cast {cid} to int: {ex}")
        except Exception as e:
            print(f"An error occurred while processing {dir_path}: {e}")

    # Remove the 0 value if present.
    all_concept_ids.discard(0)
    # Add the concept id 441840 explicitly.
    all_concept_ids.add(441840)

    all_concept_ids = sorted(list(all_concept_ids))

    result_df = pl.DataFrame({"conceptId": all_concept_ids}, schema={"conceptId": pl.Int64})

    output_folder_path = pathlib.Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_folder_path / "concept_ids.csv"
    result_df.write_csv(output_csv)
    print(f"Saved all concept ids to {output_csv}")

if __name__ == "__main__":
    main()