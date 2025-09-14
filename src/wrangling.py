import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy.stats import skew
import json
import yaml
from typing import Dict, Any, Optional


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


def create_quote_aggregations(quotes_df: pd.DataFrame, has_convert: bool = True) -> pd.DataFrame:
    """
    Crea agregaciones a nivel de cuenta desde los datos de quotes.
    
    Args:
        quotes_df: DataFrame con datos de quotes
        has_convert: Si tiene columna 'convert' para calcular account_value
    
    Returns:
        DataFrame con agregaciones por account_uuid
    """
    # Agregaciones básicas
    agg_dict = {
        'product': ['count', 'nunique'],  # num_quotes, num_products_requested
        'premium': ['sum', 'mean', 'min', 'max', lambda x: x.max() - x.min(),lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'carrier_id': lambda x: x.nunique() / len(x),  # carrier_diversity
    }
    
    quotes_agg = quotes_df.groupby("account_uuid").agg(agg_dict)
    
    # Aplanar nombres de columnas
    quotes_agg.columns = [
        'num_quotes', 'num_products_requested', 'sum_premium', 'avg_premium', 
        'min_premium', 'max_premium', 'premium_range', 'iqr_premium','carrier_diversity',
    ]
    
    # Calcular premium_ratio_max_avg
    quotes_agg['premium_ratio_max_avg'] = quotes_agg['max_premium'] / (quotes_agg['avg_premium'] + 1e-6)
    
    # Solo calcular account_value si existe columna convert
    if has_convert and 'convert' in quotes_df.columns:
        account_value = quotes_df.groupby('account_uuid').apply(
            lambda x: x.loc[x['convert'] == 1, 'premium'].sum()
        ).rename('account_value')
        quotes_agg = quotes_agg.join(account_value)
    
    quotes_agg.reset_index(inplace=True)
    return quotes_agg

def create_carrier_product_stats(quotes_df: pd.DataFrame) -> tuple:
    """Crea estadísticas de carriers y productos."""
    
    # Estadísticas de carriers
    carrier_stats = quotes_df.groupby("account_uuid")["carrier_id"].agg(
        num_carriers="nunique",
        total_quotes="count",
        carrier_concentration=lambda x: x.value_counts().max() / len(x)
    ).reset_index()
    
    # Estadísticas de productos
    product_stats = quotes_df.groupby("account_uuid")["product"].agg(
        product_concentration=lambda x: x.value_counts().max() / len(x)
    ).reset_index()
        
    return carrier_stats, product_stats


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas."""
    df = df.copy()
    
    # Features de ratio y logarítmicos
    df["premium_to_revenue_ratio"] = df["avg_premium"] / (df["annual_revenue"] + 1)
    df['log_annual_revenue'] = np.log(df['annual_revenue'] + 1)
    df['log_total_payroll'] = np.log(df['total_payroll'] + 1)
    df['revenue_per_employee'] = df['annual_revenue'] / (df['num_employees'] + 1)
    
    # Intensidad de cotizaciones
    df["quotes_per_employee"] = df["num_quotes"] / (df["num_employees"] + 1)
    df["quotes_per_million_revenue"] = df["num_quotes"] / (df["annual_revenue"]/1e6 + 1)
    
    # Premium por diferentes métricas
    df['premium_per_employee'] = df['sum_premium'] / (df['num_employees'] + 1)
    df['premium_per_revenue'] = df['sum_premium'] / (df['annual_revenue'] + 1)
    df['premium_per_quote'] = df['sum_premium'] / (df['num_quotes'] + 1)
    
    # Interacciones entre features
    df["max_x_nquotes"] = np.log(df["max_premium"]*df["num_quotes"] + 1)
    df["avg_x_nproducts"] = np.log(df["avg_premium"]*df["num_products_requested"] + 1)
    df["revenue_x_payroll"] = np.log(df["annual_revenue"] * df["total_payroll"] + 1)
    
    return df


def create_categorical_encodings(df: pd.DataFrame, encoding_stats: Optional[Dict] = None, 
                               is_training: bool = True, encodings_path: str = "config/encodings.yaml") -> tuple:
    """
    Crea encodings categóricos basados en mediana.
    
    Args:
        df: DataFrame con los datos
        encoding_stats: Estadísticas pre-calculadas (para test set)
        is_training: Si es conjunto de entrenamiento o test
        encodings_path: Ruta donde guardar/cargar los encodings
    
    Returns:
        Tuple con DataFrame procesado y estadísticas de encoding
    """
    df = df.copy()
    encodings_path = Path(encodings_path)

    if is_training:
        # Calcular estadísticas desde los datos de entrenamiento
        encoding_stats = {}
        
        # Encodings por estado
        encoding_stats['state_premium_sum'] = df.groupby('region')['sum_premium'].median().to_dict()
        encoding_stats['state_revenue'] = df.groupby('region')['annual_revenue'].median().to_dict()
        
        # Encodings por business_structure
        encoding_stats['business_structure_premium_sum'] = df.groupby('business_structure')['sum_premium'].median().to_dict()
        encoding_stats['business_structure_revenue'] = df.groupby('business_structure')['annual_revenue'].median().to_dict()
        
        # Encodings por industry
        encoding_stats['industry_premium_sum'] = df.groupby('industry')['sum_premium'].median().to_dict()
        encoding_stats['industry_revenue'] = df.groupby('industry')['annual_revenue'].median().to_dict()
        # Encodings por subindustry 
        encoding_stats['subindustry_premium_sum'] = df.groupby('subindustry')['sum_premium'].median().to_dict()
        encoding_stats['subindustry_revenue'] = df.groupby('subindustry')['annual_revenue'].median().to_dict()

        # Guardar encodings como YAML
        encodings_path.parent.mkdir(exist_ok=True)
        import yaml
        with open(encodings_path, 'w') as f:
            yaml.dump(encoding_stats, f, indent=2)
            
    else:
        # Para test: cargar encodings o usar los pasados como parámetro
        if encoding_stats is None:
            try:
                import yaml
                with open(encodings_path, 'r') as f:
                    encoding_stats = yaml.safe_load(f)
            except (FileNotFoundError, yaml.YAMLError):
                raise FileNotFoundError(f"No se pudieron cargar encodings desde {encodings_path}")

    # Aplicar encodings
    df['state_premium_sum_encoded'] = df['region'].map(encoding_stats['state_premium_sum'])
    df['state_revenue_encoded'] = df['region'].map(encoding_stats['state_revenue'])
    
    df['business_structure_premium_sum_encoded'] = df['business_structure'].map(encoding_stats['business_structure_premium_sum'])
    df['business_structure_revenue_encoded'] = df['business_structure'].map(encoding_stats['business_structure_revenue'])
    
    df['industry_sum_premium_encoded'] = df['industry'].map(encoding_stats['industry_premium_sum'])
    df['industry_revenue_encoded'] = df['industry'].map(encoding_stats['industry_revenue'])
    
    df['subindustry_sum_premium_encoded'] = df['subindustry'].map(encoding_stats['subindustry_premium_sum'])
    df['subindustry_revenue_encoded'] = df['subindustry'].map(encoding_stats['subindustry_revenue'])
    
    # Rellenar valores faltantes con medianas globales
    if is_training:
        for col in ['state_premium_sum_encoded', 'state_revenue_encoded', 
                   'business_structure_premium_sum_encoded', 'business_structure_revenue_encoded',
                   'industry_sum_premium_encoded', 'industry_revenue_encoded','subindustry_sum_premium_encoded','subindustry_revenue_encoded']:
            df[col].fillna(df[col].median(), inplace=True)
    else:
        global_defaults = {
            'state_premium_sum_encoded': np.median(list(encoding_stats['state_premium_sum'].values())),
            'state_revenue_encoded': np.median(list(encoding_stats['state_revenue'].values())),
            'business_structure_premium_sum_encoded': np.median(list(encoding_stats['business_structure_premium_sum'].values())),
            'business_structure_revenue_encoded': np.median(list(encoding_stats['business_structure_revenue'].values())),
            'industry_sum_premium_encoded': np.median(list(encoding_stats['industry_premium_sum'].values())),
            'industry_revenue_encoded': np.median(list(encoding_stats['industry_revenue'].values())),
            'subindustry_sum_premium_encoded': np.median(list(encoding_stats['subindustry_premium_sum'].values())),
            'subindustry_revenue_encoded': np.median(list(encoding_stats['subindustry_revenue'].values())),
        }
        
        for col, default_val in global_defaults.items():
            df[col].fillna(default_val, inplace=True)
    
    return df, encoding_stats


def engineer_features(accounts_df: pd.DataFrame, quotes_df: pd.DataFrame, 
                     encoding_stats: Optional[Dict] = None, is_training: bool = True) -> tuple:
    """
    Main function for feature engineering.

    Args:
        accounts_df: DataFrame con datos de accounts
        quotes_df: DataFrame con datos de quotes
        encoding_stats: Estadísticas pre-calculadas para encoding (solo para test)
        is_training: Si es conjunto de entrenamiento
    
    Returns:
        Tuple con DataFrame final y estadísticas de encoding
    """
    print("=== Starting wrangling ===")
    
    # Verify if quotes has 'convert' column
    has_convert = 'convert' in quotes_df.columns
    print(f"Column 'convert' found: {has_convert}")
    
    print("1. Creating quote aggregations...")
    quotes_agg = create_quote_aggregations(quotes_df, has_convert=has_convert)
    
    
    print("2. Creating carrier and product stats...")
    carrier_stats, product_stats = create_carrier_product_stats(quotes_df)
    
    
    print("3. Merging data...")
    df = accounts_df.merge(quotes_agg, on="account_uuid", how="left")
    df = df.merge(carrier_stats, on="account_uuid", how="left")
    df = df.merge(product_stats, on="account_uuid", how="left") 
    
    print("4. Creating features...")
    df = create_derived_features(df)
    
    
    print("5. Creating categorical encodings...")
    df, encoding_stats_result = create_categorical_encodings(df, encoding_stats, is_training)
    
    print(f"=== Wrangling completed. Final shape: {df.shape} ===")
    
    return df, encoding_stats_result


def process_dataset(config: Dict, dataset_name: str, encoding_stats: Optional[Dict] = None) -> tuple:
    """
    Process an specific dataset (train or test).
    Args:
        config: Configuration
        dataset_name: Name of the dataset ('train' o 'test')
        encoding_stats: Stats of enconding of train dataset (REQUIRED for test)
    
    Returns:
        Tuple with DataFrame processed and encoding stats.
    """
    BASE_DIR = Path.cwd().parent
    data_path = BASE_DIR / config.get('data_dir', 'data')
    
    dataset_config = config['datasets'][dataset_name]
    
    # Cargar datos
    print(f"\n=== Procesando dataset: {dataset_name} ===")
    print(f"Cargando {dataset_config['accounts_file']}...")
    accounts = pd.read_csv(data_path / dataset_config['accounts_file'])
    
    print(f"Cargando {dataset_config['quotes_file']}...")
    quotes = pd.read_csv(data_path / dataset_config['quotes_file'])
    
    # Verificar prerequisitos para test
    is_training = (dataset_name == 'train')
    if not is_training and encoding_stats is None:
        raise ValueError(
            f"Para procesar el dataset '{dataset_name}' (test), "
            "debe proporcionar encoding_stats del conjunto de entrenamiento. "
            "Procese primero el conjunto 'train' y pase sus encoding_stats."
        )
    
    # Procesar features
    df, encoding_stats_result = engineer_features(
        accounts, quotes, encoding_stats, is_training=is_training
    )
    
    # Guardar resultado
    output_path = data_path / dataset_config['output_file']
    print(f"Guardando resultado en {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"Procesamiento de {dataset_name} completado.")
    return df, encoding_stats_result


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Feature engineering for accounts and quotes data')
    
    parser.add_argument('--config', required=True,
                       help='Config file (JSON o YAML)')
    parser.add_argument('--dataset', 
                       help='Specific dataset to process (train/test). If not specified, process both.')
    parser.add_argument('--encoding-stats-path', 
                       help='Path of the encoding stats (default: data/encoding_stats.json)')
    
    args = parser.parse_args()
    
    try:
        # Cargar configuración
        config = load_config(args.config)
        
        # Determinar ruta de encoding_stats
        if args.encoding_stats_path:
            encoding_stats_path = args.encoding_stats_path
        else:
            base_path = Path(config.get('base_dir', '.'))
            data_path = base_path / config.get('config_dir', 'config')
            encoding_stats_path = str(data_path / 'encoding_stats.json')
            print(encoding_stats_path)
        
        encoding_stats = None  # Variable para almacenar stats de train
        
        if args.dataset:
            # Procesar dataset específico
            if args.dataset not in config['datasets']:
                raise ValueError(f"Dataset '{args.dataset}' no encontrado en configuración")
            
            # Para test, cargar encoding_stats si existen
            if args.dataset == 'test':
                try:
                    import json
                    with open(encoding_stats_path, 'r') as f:
                        encoding_stats = json.load(f)
                except FileNotFoundError:
                    raise ValueError(f"No se encontraron encoding_stats en {encoding_stats_path}. "
                                   "Ejecute primero el dataset 'train'.")
            
            df, encoding_stats_result = process_dataset(
                config, args.dataset, encoding_stats=encoding_stats
            )
            
            # Guardar encoding_stats si es train
            if args.dataset == 'train':
                import json
                with open(encoding_stats_path, 'w') as f:
                    json.dump(encoding_stats_result, f, indent=2)
                print(f"📊 Encoding stats guardados en: {encoding_stats_path}")
            
            print(f"\n✅ Dataset '{args.dataset}' procesado exitosamente")
        
        else:
            # Procesar ambos datasets (primero train, luego test)
            datasets_to_process = ['train', 'test']
            
            for dataset_name in datasets_to_process:
                if dataset_name in config['datasets']:
                    print(f"\n{'='*50}")
                    print(f"Procesando {dataset_name.upper()}")
                    print('='*50)
                    
                    df, encoding_stats_result = process_dataset(
                        config, dataset_name, encoding_stats=encoding_stats
                    )
                    
                    if dataset_name == 'train':
                        # Guardar encoding_stats para usar en test
                        encoding_stats = encoding_stats_result
                        import json
                        with open(encoding_stats_path, 'w') as f:
                            json.dump(encoding_stats_result, f, indent=2)
                        print(f"📊 Encoding stats guardados en: {encoding_stats_path}")
                        print("🔗 Estos stats se usarán automáticamente para procesar test")
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