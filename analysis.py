import pandas as pd
import numpy as np
from scipy import stats
import logging
from datetime import datetime, timedelta
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class BacktestAnalyzer:
    def __init__(self, capital_inicial=2500):
        """
        Inicializa el analizador de backtest.
        
        Args:
            capital_inicial (float): Capital inicial en USD
        """
        self.capital_inicial = capital_inicial
        self.symbol = 'BTCUSDT'
        os.makedirs('results', exist_ok=True)
        logging.info("BacktestAnalyzer inicializado")
        
    def cargar_resultados(self):
        """
        Carga los resultados del backtest.
        
        Returns:
            tuple: (DataFrame operaciones, DataFrame balance)
        """
        try:
            # Intentar cargar archivos de resultados
            ops_file = f"results/{self.symbol}-results.csv"
            balance_file = f"results/{self.symbol}-balance.csv"
            
            if not os.path.exists(ops_file) or not os.path.exists(balance_file):
                logging.warning("Archivos de resultados no encontrados")
                return self._crear_archivo_resultados_vacio()
                
            # Cargar operaciones
            df_ops = pd.read_csv(ops_file)
            df_ops['fecha_entrada'] = pd.to_datetime(df_ops['fecha_entrada'])
            df_ops['fecha_salida'] = pd.to_datetime(df_ops['fecha_salida'])
            
            # Cargar balance
            df_balance = pd.read_csv(balance_file)
            df_balance['fecha'] = pd.to_datetime(df_balance['fecha'])
            df_balance.set_index('fecha', inplace=True)
            
            logging.info(f"Resultados cargados: {len(df_ops)} operaciones")
            return df_ops, df_balance
            
        except Exception as e:
            logging.error(f"Error al cargar resultados: {str(e)}")
            return self._crear_archivo_resultados_vacio()
            
    def calcular_metricas_rendimiento(self, df_ops):
        """
        Calcula métricas de rendimiento del backtest.
        
        Args:
            df_ops (pandas.DataFrame): DataFrame de operaciones
            
        Returns:
            dict: Métricas de rendimiento
        """
        try:
            if df_ops.empty:
                raise ValueError("No hay operaciones para analizar")
                
            # Métricas básicas
            total_ops = len(df_ops)
            wins = len(df_ops[df_ops['resultado'] == 'TP'])
            losses = len(df_ops[df_ops['resultado'] == 'SL'])
            
            # Win rates
            win_rate_total = (wins / total_ops * 100) if total_ops > 0 else 0
            
            # Win rates por dirección
            longs = df_ops[df_ops['direccion'] == 'LONG']
            shorts = df_ops[df_ops['direccion'] == 'SHORT']
            
            win_rate_long = (
                len(longs[longs['resultado'] == 'TP']) / len(longs) * 100 
                if len(longs) > 0 else 0
            )
            win_rate_short = (
                len(shorts[shorts['resultado'] == 'TP']) / len(shorts) * 100 
                if len(shorts) > 0 else 0
            )
            
            # Análisis de patrones
            confianza_media = df_ops['confianza_patron'].mean()
            
            # Efectividad de patrones
            ops_alta_confianza = df_ops[df_ops['confianza_patron'] >= 0.7]
            if len(ops_alta_confianza) > 0:
                wins_alta_confianza = len(ops_alta_confianza[
                    ops_alta_confianza['resultado'] == 'TP'
                ])
                efectividad = wins_alta_confianza / len(ops_alta_confianza)
            else:
                efectividad = 0
                
            # Retornos
            retorno_medio = df_ops['retorno_operacion'].mean()
            retorno_std = df_ops['retorno_operacion'].std()
            
            # Factor de beneficio
            gains = df_ops[df_ops['retorno_operacion'] > 0]['retorno_operacion'].sum()
            losses = abs(df_ops[df_ops['retorno_operacion'] < 0]['retorno_operacion'].sum())
            profit_factor = gains / losses if losses != 0 else float('inf')
            
            return {
                'total_operaciones': total_ops,
                'operaciones_ganadoras': wins,
                'operaciones_perdedoras': losses,
                'win_rate_total': win_rate_total,
                'win_rate_long': win_rate_long,
                'win_rate_short': win_rate_short,
                'retorno_medio': retorno_medio,
                'retorno_std': retorno_std,
                'profit_factor': profit_factor,
                'confianza_media_patrones': confianza_media,
                'efectividad_patrones': efectividad
            }
            
        except Exception as e:
            logging.error(f"Error al calcular métricas de rendimiento: {str(e)}")
            return {}
            
    def calcular_ratios(self, df_ops, df_balance):
        """
        Calcula ratios financieros del backtest.
        
        Args:
            df_ops (pandas.DataFrame): DataFrame de operaciones
            df_balance (pandas.DataFrame): DataFrame de balance
            
        Returns:
            dict: Ratios financieros
        """
        try:
            if df_ops.empty or df_balance.empty:
                raise ValueError("No hay datos suficientes para calcular ratios")
                
            # Calcular retornos diarios
            retornos_diarios = df_balance['balance'].pct_change().dropna()
            
            # Sharpe Ratio (asumiendo tasa libre de riesgo = 0)
            sharpe_ratio = np.sqrt(252) * (
                retornos_diarios.mean() / retornos_diarios.std()
            ) if retornos_diarios.std() != 0 else 0
            
            # Sortino Ratio
            retornos_negativos = retornos_diarios[retornos_diarios < 0]
            downside_std = retornos_negativos.std()
            sortino_ratio = np.sqrt(252) * (
                retornos_diarios.mean() / downside_std
            ) if downside_std != 0 else 0
            
            # Calmar Ratio
            drawdown = self.calcular_drawdown(df_balance)
            max_dd = drawdown['max_drawdown']
            retorno_anualizado = (
                (df_balance['balance'].iloc[-1] / self.capital_inicial) 
                ** (252 / len(df_balance)) - 1
            )
            calmar_ratio = abs(retorno_anualizado / max_dd) if max_dd != 0 else 0
            
            # Profit Factor
            gains = df_ops[df_ops['retorno_operacion'] > 0]['retorno_operacion'].sum()
            losses = abs(df_ops[df_ops['retorno_operacion'] < 0]['retorno_operacion'].sum())
            profit_factor = gains / losses if losses != 0 else float('inf')
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logging.error(f"Error al calcular ratios: {str(e)}")
            return {}
            
    def calcular_drawdown(self, df_balance):
        """
        Calcula métricas de drawdown.
        
        Args:
            df_balance (pandas.DataFrame): DataFrame de balance
            
        Returns:
            dict: Métricas de drawdown
        """
        try:
            if df_balance.empty:
                raise ValueError("No hay datos de balance para calcular drawdown")
                
            # Calcular drawdown
            balance = df_balance['balance'].values
            peak = np.maximum.accumulate(balance)
            drawdown = (balance - peak) / peak
            
            # Encontrar máximo drawdown
            max_dd = drawdown.min()
            max_dd_idx = np.argmin(drawdown)
            
            # Encontrar el pico anterior al drawdown máximo
            peak_idx = np.argmax(balance[:max_dd_idx])
            
            # Calcular duración
            dd_duration = df_balance.index[max_dd_idx] - df_balance.index[peak_idx]
            
            # Calcular retorno total
            retorno_total = ((balance[-1] - self.capital_inicial) / 
                           self.capital_inicial) * 100
            
            return {
                'max_drawdown': max_dd,
                'drawdown_duration': dd_duration,
                'balance_final': balance[-1],
                'retorno_total': retorno_total
            }
            
        except Exception as e:
            logging.error(f"Error al calcular drawdown: {str(e)}")
            return {}
            
    def _crear_archivo_resultados_vacio(self):
        """
        Crea archivos de resultados vacíos.
        
        Returns:
            tuple: (DataFrame operaciones vacío, DataFrame balance vacío)
        """
        try:
            columnas_ops = [
                'fecha_entrada', 'fecha_salida', 'direccion', 'precio_entrada',
                'precio_salida', 'stop_loss', 'take_profit', 'resultado',
                'retorno_operacion', 'confianza_patron'
            ]
            df_ops = pd.DataFrame(columns=columnas_ops)
            
            columnas_balance = ['fecha', 'balance']
            df_balance = pd.DataFrame(columns=columnas_balance)
            df_balance.set_index('fecha', inplace=True)
            
            return df_ops, df_balance
            
        except Exception as e:
            logging.error(f"Error al crear archivos vacíos: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

def main():
    """Función principal para pruebas"""
    try:
        analyzer = BacktestAnalyzer()
        df_ops, df_balance = analyzer.cargar_resultados()
        
        if df_ops.empty or df_balance.empty:
            logging.warning("No hay datos para analizar")
            return
            
        metricas = analyzer.calcular_metricas_rendimiento(df_ops)
        ratios = analyzer.calcular_ratios(df_ops, df_balance)
        drawdown = analyzer.calcular_drawdown(df_balance)
        
        print("\nMétricas de rendimiento:")
        for k, v in metricas.items():
            print(f"{k}: {v}")
            
        print("\nRatios:")
        for k, v in ratios.items():
            print(f"{k}: {v}")
            
        print("\nDrawdown:")
        for k, v in drawdown.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()