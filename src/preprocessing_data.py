from utils import *
import pandas as pd
import argparse
import os
from pathlib import Path
import json
import yaml
from typing import Dict, Any
from utils import detect_base_path



def load_config(config_path: str) -> Dict[str, Any]:
    """Carga configuración desde archivo JSON o YAML."""
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.json':
            return json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Formato de archivo no soportado: {config_path.suffix}")


def aggregate_quotes(quotes_df, has_convert_column=True):
    """
    Aggregating data from quotes by account_uuid.
    
    Args:
        quotes_df: DataFrame with quotes data
        has_convert_column: If the dataframe has the 'convert' column to calculate account_value
    
    Returns:
        DataFrame aggregated by account_uuid with metrics
    """
    agg_dict = {
        'product': 'count',  
    }
    
    # Only calculate account_value if 'convert' column exists
    if has_convert_column and 'convert' in quotes_df.columns:
        agg_dict['premium'] = lambda x: x[quotes_df.loc[x.index, 'convert'] == 1].sum()
        columns = ['account_uuid', 'account_value', 'total_quoted_products']
    else:
        columns = ['account_uuid', 'total_quoted_products']
    
    quotes_agg = quotes_df.groupby('account_uuid').agg(agg_dict)
    quotes_agg.reset_index(inplace=True)
    quotes_agg.columns = columns
    
    return quotes_agg


def fill_missing_values(data):
    """
    Filling the null values according to the defined strategy.
    
    Args:
        data: DataFrame to process
    
    Returns:
        DataFrame with filled null values
    """
    data_copy = data.copy()
    
    # Filling state and business_structure with the mode (low % of nulls)
    if 'state' in data_copy.columns:
        dic_states = {
            'California': 'CA',
            'New York': 'NY',
            'PA - Pennsylvania': 'PA',
            'Washington DC': 'DC',
            'Oregon': 'OR',
            'Florida': 'FL'
        }

        data_copy['state'] = data_copy['state'].map(dic_states).fillna(data_copy['state'])
        state_to_region = {
            # NORTHEAST (9 estados + DC)
            'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
            'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast', 'RI': 'Northeast', 
            'VT': 'Northeast', 'DC': 'Northeast',

            # MIDWEST (12 estados)
            'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest',
            'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
            'ND': 'Midwest', 'OH': 'Midwest', 'SD': 'Midwest', 'WI': 'Midwest',

            # SOUTH (16 estados + DC ya incluido arriba)
            'AL': 'South', 'AR': 'South', 'DE': 'South', 'FL': 'South',
            'GA': 'South', 'KY': 'South', 'LA': 'South', 'MD': 'South',
            'MS': 'South', 'NC': 'South', 'OK': 'South', 'SC': 'South',
            'TN': 'South', 'TX': 'South', 'VA': 'South', 'WV': 'South',

            # WEST (13 estados)
            'AK': 'West', 'AZ': 'West', 'CA': 'West', 'CO': 'West',
            'HI': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
            'NM': 'West', 'OR': 'West', 'UT': 'West', 'WA': 'West', 'WY': 'West'
        }

        # Crear columna de región
        data_copy['region'] = data_copy['state'].map(state_to_region)
        data_copy['state'].fillna(data_copy['state'].mode()[0], inplace=True)
    
    if 'business_structure' in data_copy.columns:
        data_copy['business_structure'].fillna(data_copy['business_structure'].mode()[0], inplace=True)
    
    # Filling industry and subindustry with 'Unknown' (higher % of nulls)
    if 'industry' in data_copy.columns:
        data_copy['industry'].fillna('Unknown', inplace=True)
    
    if 'subindustry' in data_copy.columns:
        data_copy['subindustry'].fillna('Unknown', inplace=True)
    
    # Filling numeric columns with group median
    numeric_cols = ['num_employees', 'total_payroll', 'annual_revenue', 'year_established']
    existing_numeric_cols = [col for col in numeric_cols if col in data_copy.columns]
    
    grouping_cols = []
    for col in ['state', 'business_structure', 'industry']:
        if col in data_copy.columns:
            grouping_cols.append(col)
    
    if grouping_cols and existing_numeric_cols:
        for col in existing_numeric_cols:
            global_median = data_copy[col].median()
            data_copy[col] = data_copy.groupby(grouping_cols)[col].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median)
            )
    
    return data_copy


def process_dataset(config: Dict, dataset_name: str) -> pd.DataFrame:
    """
    Process a specific dataset (train or test).
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset ('train' or 'test')
    
    Returns:
        Processed DataFrame
    """
    # CAMBIO: Usar función de detección en lugar de Path.cwd().parent
    BASE_DIR = detect_base_path()
    data_path = BASE_DIR / config.get('data_dir', 'data')
    
    dataset_config = config['datasets'][dataset_name]
    
    # Cargar datos
    print(f"\n=== Procesando dataset: {dataset_name} ===")
    print(f"Cargando {dataset_config['accounts_file']}...")
    accounts = pd.read_csv(data_path / dataset_config['accounts_file'])
    
    print(f"Cargando {dataset_config['quotes_file']}...")
    quotes = pd.read_csv(data_path / dataset_config['quotes_file'])
    
    has_convert = 'convert' in quotes.columns
    print(f"Column 'convert' found: {has_convert}")
    
    print("Aggregating quotes...")
    quotes_agg = aggregate_quotes(quotes, has_convert_column=has_convert)
    
    print("Merging account and quotes...")
    data = accounts.merge(quotes_agg, on='account_uuid', how='left')
    
    print("Filling null values...")
    data = fill_missing_values(data)
    
    # Drop columns that won't be used for modeling
    columns_to_drop = []
    if 'total_quoted_products' in data.columns:
        columns_to_drop.append('total_quoted_products')
    if 'account_value' in data.columns:
        columns_to_drop.append('account_value')
    
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
    
    # Guardar resultado
    output_path = data_path / dataset_config['output_file']
    print(f"Saving the account_data in {output_path}...")
    data.to_csv(output_path, index=False)
    
    print(f"Preprocessing completed.")
    print(f"Final shape: {data.shape}")
    
    return data


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Preprocessing de datos de accounts y quotes')
    
    parser.add_argument('--config', required=True,
                       help='Config file (JSON o YAML)')
    parser.add_argument('--dataset', 
                       help='Specific dataset to process (train/test). If not specified, process both.')
    
    args = parser.parse_args()

    try:
        # CAMBIO: Manejar ruta de config de forma inteligente
        base_path = detect_base_path()
        
        # Inicializar config_path desde el argumento
        config_path = Path(args.config)

        if not config_path.is_absolute():
            # Si no es absoluta, buscar desde base_path/src
            if (base_path / 'src' / args.config).exists():
                config_path = base_path / 'src' / args.config
            elif not config_path.exists():
                config_path = base_path / args.config

        config = load_config(str(config_path))
        
        if args.dataset:
            # Procesar dataset específico
            if args.dataset not in config['datasets']:
                raise ValueError(f"Dataset '{args.dataset}' no encontrado en configuración")
            
            df = process_dataset(config, args.dataset)
            print(f"\n✅ Dataset '{args.dataset}' procesado exitosamente")
        
        else:
            # Procesar ambos datasets
            datasets_to_process = ['train', 'test']
            
            for dataset_name in datasets_to_process:
                if dataset_name in config['datasets']:
                    print(f"\n{'='*50}")
                    print(f"Procesando {dataset_name.upper()}")
                    print('='*50)
                    
                    df = process_dataset(config, dataset_name)
                else:
                    print(f"⚠️  Dataset '{dataset_name}' no encontrado en configuración, omitiendo...")
        
        print("\n🎉 Procesamiento completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        import traceback
        print("\n📋 Detalles del error:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())