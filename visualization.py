import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class BacktestVisualizer:
    def __init__(self, analyzer):
        """
        Inicializa el visualizador de backtest.
        
        Args:
            analyzer: Instancia de BacktestAnalyzer
        """
        self.analyzer = analyzer
        self.symbol = 'BTCUSDT'
        
        # Configuración de estilo
        plt.style.use('default')
        self.colors = {
            'LONG': '#2ecc71',
            'SHORT': '#e74c3c',
            'balance': '#3498db',
            'drawdown': '#e74c3c',
            'pattern': '#9b59b6'
        }
        
        # Crear directorio para gráficos
        os.makedirs('figures', exist_ok=True)
        os.makedirs('figures/patterns', exist_ok=True)
        logging.info("BacktestVisualizer inicializado")
        
    def generar_graficos(self, df_ops, df_balance):
        """
        Genera todos los gráficos del backtest.
        
        Args:
            df_ops (pandas.DataFrame): DataFrame de operaciones
            df_balance (pandas.DataFrame): DataFrame de balance
        """
        try:
            if df_ops.empty or df_balance.empty:
                raise ValueError("No hay datos suficientes para generar gráficos")
                
            logging.info("Iniciando generación de gráficos")
            
            # Gráficos principales
            self.graficar_balance(df_balance)
            self.graficar_drawdown(df_balance)
            self.graficar_retornos_acumulados(df_ops)
            self.graficar_distribucion_retornos(df_ops)
            self.graficar_matriz_correlacion(df_ops)
            self.graficar_confianza_patrones(df_ops)
            
            logging.info("Gráficos generados exitosamente")
            
        except Exception as e:
            logging.error(f"Error al generar gráficos: {str(e)}")
            raise
            
    def graficar_balance(self, df_balance):
        """
        Genera gráfico de evolución del balance.
        
        Args:
            df_balance (pandas.DataFrame): DataFrame de balance
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df_balance.index, df_balance['balance'], 
                    color=self.colors['balance'], linewidth=2)
            
            plt.title('Evolución del Balance', fontsize=12, pad=15)
            plt.xlabel('Fecha')
            plt.ylabel('Balance (USD)')
            plt.grid(True, alpha=0.3)
            
            # Añadir capital inicial como línea horizontal
            plt.axhline(y=self.analyzer.capital_inicial, color='gray', 
                       linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig('figures/balance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al graficar balance: {str(e)}")
            
    def graficar_drawdown(self, df_balance):
        """
        Genera gráfico de drawdown.
        
        Args:
            df_balance (pandas.DataFrame): DataFrame de balance
        """
        try:
            # Calcular drawdown
            balance = df_balance['balance'].values
            peak = np.maximum.accumulate(balance)
            drawdown = (balance - peak) / peak
            
            plt.figure(figsize=(12, 6))
            plt.plot(df_balance.index, drawdown * 100, 
                    color=self.colors['drawdown'], linewidth=2)
            
            plt.title('Drawdown (%)', fontsize=12, pad=15)
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # Añadir línea de 0%
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig('figures/drawdown.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al graficar drawdown: {str(e)}")
            
    def graficar_retornos_acumulados(self, df_ops):
        """
        Genera gráfico de retornos acumulados.
        
        Args:
            df_ops (pandas.DataFrame): DataFrame de operaciones
        """
        try:
            retornos = df_ops['retorno_operacion'].values
            retornos_acum = np.cumsum(retornos)
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(retornos_acum)), retornos_acum * 100, 
                    color=self.colors['balance'], linewidth=2)
            
            plt.title('Retornos Acumulados (%)', fontsize=12, pad=15)
            plt.xlabel('Número de Operaciones')
            plt.ylabel('Retorno Acumulado (%)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('figures/returns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al graficar retornos acumulados: {str(e)}")
            
    def graficar_distribucion_retornos(self, df_ops):
        """
        Genera gráfico de distribución de retornos.
        
        Args:
            df_ops (pandas.DataFrame): DataFrame de operaciones
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Separar retornos por dirección
            retornos_long = df_ops[df_ops['direccion'] == 'LONG']['retorno_operacion']
            retornos_short = df_ops[df_ops['direccion'] == 'SHORT']['retorno_operacion']
            
            # Crear histograma
            plt.hist(retornos_long * 100, bins=30, alpha=0.5, 
                    color=self.colors['LONG'], label='LONG')
            plt.hist(retornos_short * 100, bins=30, alpha=0.5, 
                    color=self.colors['SHORT'], label='SHORT')
            
            plt.title('Distribución de Retornos por Dirección', fontsize=12, pad=15)
            plt.xlabel('Retorno (%)')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('figures/returns_distribution.png', dpi=300, 
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al graficar distribución de retornos: {str(e)}")
            
    def graficar_matriz_correlacion(self, df_ops):
        """
        Genera matriz de correlación entre variables clave.
        
        Args:
            df_ops (pandas.DataFrame): DataFrame de operaciones
        """
        try:
            # Preparar datos para correlación
            data_corr = pd.DataFrame({
                'retorno': df_ops['retorno_operacion'],
                'confianza': df_ops['confianza_patron'],
                'volatilidad': df_ops['precio_salida'] / df_ops['precio_entrada'] - 1
            })
            
            # Calcular matriz de correlación
            corr_matrix = data_corr.corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True)
            
            plt.title('Matriz de Correlación', fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig('figures/correlation_matrix.png', dpi=300, 
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al graficar matriz de correlación: {str(e)}")
            
    def graficar_confianza_patrones(self, df_ops):
        """
        Genera gráfico de relación entre confianza y retornos.
        
        Args:
            df_ops (pandas.DataFrame): DataFrame de operaciones
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.scatter(df_ops['confianza_patron'], 
                       df_ops['retorno_operacion'] * 100,
                       c=df_ops['retorno_operacion'].apply(
                           lambda x: self.colors['LONG'] if x > 0 
                           else self.colors['SHORT']
                       ),
                       alpha=0.6)
            
            plt.title('Relación Confianza-Retorno', fontsize=12, pad=15)
            plt.xlabel('Confianza del Patrón')
            plt.ylabel('Retorno (%)')
            plt.grid(True, alpha=0.3)
            
            # Añadir línea de tendencia
            z = np.polyfit(df_ops['confianza_patron'], 
                          df_ops['retorno_operacion'] * 100, 1)
            p = np.poly1d(z)
            plt.plot(df_ops['confianza_patron'], 
                    p(df_ops['confianza_patron']), 
                    color='gray', linestyle='--', alpha=0.8)
            
            plt.tight_layout()
            plt.savefig('figures/pattern_confidence.png', dpi=300, 
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al graficar confianza de patrones: {str(e)}")
            
    def visualizar_patron(self, precios, patron_info, idx_actual):
        """
        Genera visualización de un patrón específico.
        
        Args:
            precios (numpy.array): Array de precios
            patron_info (dict): Información del patrón
            idx_actual (int): Índice actual
        """
        try:
            window_size = patron_info['window_size']
            start_idx = patron_info['start_idx']
            end_idx = patron_info['end_idx']
            
            plt.figure(figsize=(12, 6))
            
            # Graficar patrón histórico
            plt.plot(range(window_size), 
                    precios[start_idx:end_idx], 
                    color=self.colors['pattern'], 
                    label='Patrón Histórico', 
                    linewidth=2)
            
            # Graficar patrón actual
            plt.plot(range(window_size), 
                    precios[idx_actual-window_size:idx_actual], 
                    color=self.colors['balance'], 
                    label='Patrón Actual', 
                    linewidth=2, 
                    linestyle='--')
            
            plt.title(f'Comparación de Patrones (Confianza: '
                     f'{patron_info["confidence"]:.2f})', 
                     fontsize=12, pad=15)
            plt.xlabel('Períodos')
            plt.ylabel('Precio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Guardar gráfico
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'figures/patterns/pattern_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Visualización de patrón guardada: {filename}")
            
        except Exception as e:
            logging.error(f"Error al visualizar patrón: {str(e)}")

def main():
    """Función principal para pruebas"""
    try:
        from analysis import BacktestAnalyzer
        
        analyzer = BacktestAnalyzer()
        df_ops, df_balance = analyzer.cargar_resultados()
        
        if df_ops.empty or df_balance.empty:
            logging.warning("No hay datos suficientes para generar visualizaciones")
            return
            
        visualizer = BacktestVisualizer(analyzer)
        visualizer.generar_graficos(df_ops, df_balance)
        
        print("Visualizaciones generadas exitosamente")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()