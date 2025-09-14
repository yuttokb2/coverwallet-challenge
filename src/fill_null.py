def impute_num_employees(df):
    """
    Número de empleados - usar business_structure como guía
    """
    # Estrategia: Mediana por business_structure
    # (Individual típicamente tiene 0-1, Corporation tiene más)
    
    median_by_structure = df.groupby('business_structure')['num_employees'].median()
    
    # Imputar
    for idx in df[df['num_employees'].isna()].index:
        structure = df.loc[idx, 'business_structure']
        df.loc[idx, 'num_employees'] = median_by_structure.get(structure, 0)
    
    # Casos especiales
    # Si es "Individual" y tiene nulo → probablemente 0
    df.loc[
        (df['num_employees'].isna()) & 
        (df['business_structure'] == 'Individual'), 
        'num_employees'
    ] = 0
    
    # Fallback: mediana global
    df['num_employees'].fillna(df['num_employees'].median(), inplace=True)
    
    return df