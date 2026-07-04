import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIG, TAB, TARGET_STATES
from src.logging_utils import Timer

def run(df, log):
    summary = {}
    
    # Try to guess column names
    cols = [c.lower() for c in df.columns]
    
    state_col = next((c for c in df.columns if 'state' in c.lower()), None)
    date_col = next((c for c in df.columns if 'datetime' in c.lower() or 'date' in c.lower()), None)
    well_col = next((c for c in df.columns if 'station' in c.lower() or 'well' in c.lower()), None)
    level_col = next((c for c in df.columns if 'target' in c.lower() or 'level' in c.lower() or 'depth' in c.lower()), None)
    
    log.info(f"Inferred columns: State='{state_col}', Date='{date_col}', Well='{well_col}', Level='{level_col}'")
    
    if date_col:
        with Timer(log, "EDA: Date Parsing"):
            try:
                df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
                df['year'] = df['parsed_date'].dt.year
                df['month'] = df['parsed_date'].dt.month
            except:
                df['year'] = df[date_col]
                df['month'] = 1
                
    with Timer(log, "EDA: Coverage"):
        if well_col:
            summary['unique_wells'] = df[well_col].nunique()
        if date_col and 'year' in df.columns:
            summary['date_range'] = f"{df['year'].min()} to {df['year'].max()}"
            
            records_per_year = df['year'].value_counts().sort_index()
            records_per_year.to_csv(TAB / "records_per_year.csv")
            summary['records_per_year_shape'] = records_per_year.shape
            
            plt.figure(figsize=(10,5))
            records_per_year.plot(kind='bar')
            plt.title('Records per Year')
            plt.tight_layout()
            plt.savefig(FIG / 'coverage_records_per_year.png', dpi=150)
            plt.close()

    with Timer(log, "EDA: Geography"):
        if state_col:
            state_counts = df[state_col].value_counts()
            state_counts.to_csv(TAB / "records_by_state.csv")
            summary['records_by_state_shape'] = state_counts.shape
            
            actual_states = df[state_col].dropna().unique()
            target_mapped = [s for s in actual_states if isinstance(s, str) and any(t.lower() in s.lower() for t in TARGET_STATES)]
            if not target_mapped:
                target_mapped = actual_states # If they are all encoded ints, just use all of them for now
            
            df_target = df[df[state_col].isin(target_mapped)]
            summary['target_region_size'] = len(df_target)
            log.info(f"Target region ({target_mapped}) subset size: {len(df_target)}")
        else:
            df_target = df
            
    with Timer(log, "EDA: Missingness"):
        missing = df.isnull().mean() * 100
        missing.to_csv(TAB / "missingness_per_column.csv")
        summary['missingness_shape'] = missing.shape
        
        plt.figure(figsize=(10,6))
        missing.plot(kind='bar')
        plt.title('% Missing per Column')
        plt.tight_layout()
        plt.savefig(FIG / 'missingness_per_column.png', dpi=150)
        plt.close()
        
        if level_col and 'year' in df.columns:
            missing_per_year = df.groupby('year')[level_col].apply(lambda x: x.isnull().sum())
            missing_per_year.to_csv(TAB / "missing_level_per_year.csv")
            
    with Timer(log, "EDA: Depletion Signal"):
        if level_col and 'year' in df_target.columns:
            yearly_mean = df_target.groupby('year')[level_col].mean().dropna()
            if not yearly_mean.empty:
                yearly_mean.to_csv(TAB / "target_yearly_mean_level.csv")
                
                plt.figure(figsize=(10,5))
                yearly_mean.plot(kind='line', marker='o')
                
                x = np.arange(len(yearly_mean))
                y = yearly_mean.values
                if len(x) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    plt.plot(yearly_mean.index, slope*x + intercept, 'r--', label=f'Trend: {slope:.3f}/yr')
                    summary['depletion_trend_slope'] = slope
                    
                plt.title(f'Mean Groundwater Level over Time ({TARGET_STATES})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(FIG / 'depletion_signal.png', dpi=150)
                plt.close()
                
    with Timer(log, "EDA: Seasonality"):
        if level_col and 'month' in df_target.columns:
            plt.figure(figsize=(10,5))
            sns.boxplot(x='month', y=level_col, data=df_target)
            plt.title('Monthly Climatology of Water Level')
            plt.tight_layout()
            plt.savefig(FIG / 'seasonality_monthly.png', dpi=150)
            plt.close()
            
    return summary
