import logging
from datetime import datetime
import os
from backtesting import Backtester
from analysis import BacktestAnalyzer
from visualization import BacktestVisualizer
from report_generator import ReportGenerator

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class TradingSystem:
    def __init__(self):
        """
        Inicializa el sistema de trading con todos los componentes.
        """
        try:
            # Crear directorios necesarios
            self._crear_directorios()
            
            # Inicializar componentes
            self.backtester = Backtester()
            self.analyzer = BacktestAnalyzer()
            self.visualizer = BacktestVisualizer(self.analyzer)
            self.report_generator = ReportGenerator()
            
            logging.info("Sistema de trading inicializado correctamente")
            
        except Exception as e:
            logging.error(f"Error en inicialización del sistema: {str(e)}")
            raise
            
    def _crear_directorios(self):
        """Crea los directorios necesarios para el sistema."""
        directorios = ['data', 'results', 'figures', 'reports', 'figures/patterns']
        for directorio in directorios:
            os.makedirs(directorio, exist_ok=True)
            
    def ejecutar_analisis_completo(self):
        """
        Ejecuta el análisis completo del sistema de trading.
        
        Returns:
            dict: Resumen de resultados
        """
        try:
            logging.info("Iniciando análisis completo del sistema")
            
            # Ejecutar backtest
            resultados, balance, fechas = self.backtester.ejecutar_backtest()
            
            if not resultados or not balance:
                return {
                    'error': 'No se generaron resultados en el backtest',
                    'status': False
                }
                
            # Cargar resultados para análisis
            df_ops, df_balance = self.analyzer.cargar_resultados()
            
            if df_ops.empty or df_balance.empty:
                return {
                    'error': 'Error al cargar resultados para análisis',
                    'status': False
                }
                
            # Calcular métricas
            metricas = self.analyzer.calcular_metricas_rendimiento(df_ops)
            ratios = self.analyzer.calcular_ratios(df_ops, df_balance)
            drawdown = self.analyzer.calcular_drawdown(df_balance)
            
            # Generar visualizaciones
            self.visualizer.generar_graficos(df_ops, df_balance)
            
            # Generar reporte
            self.report_generator.generar_reporte(
                metricas, ratios, drawdown, df_ops
            )
            
            # Preparar resumen
            resumen = self._preparar_resumen(
                metricas, ratios, drawdown, len(resultados)
            )
            
            logging.info("Análisis completo finalizado exitosamente")
            return resumen
            
        except Exception as e:
            logging.error(f"Error en ejecución del análisis: {str(e)}")
            return {'error': str(e), 'status': False}
            
    def _preparar_resumen(self, metricas, ratios, drawdown, total_ops):
        """
        Prepara un resumen ejecutivo de los resultados.
        
        Args:
            metricas (dict): Métricas de rendimiento
            ratios (dict): Ratios financieros
            drawdown (dict): Métricas de drawdown
            total_ops (int): Total de operaciones
            
        Returns:
            dict: Resumen ejecutivo
        """
        try:
            return {
                'status': True,
                'metricas_principales': {
                    'Total Operaciones': total_ops,
                    'Win Rate Total': f"{metricas['win_rate_total']:.1f}%",
                    'Win Rate Long': f"{metricas['win_rate_long']:.1f}%",
                    'Win Rate Short': f"{metricas['win_rate_short']:.1f}%",
                    'Profit Factor': f"{ratios['profit_factor']:.2f}",
                    'Sharpe Ratio': f"{ratios['sharpe_ratio']:.2f}",
                    'Max Drawdown': f"{drawdown['max_drawdown']*100:.1f}%",
                    'Retorno Total': f"{drawdown['retorno_total']:.1f}%"
                },
                'patrones': {
                    'Efectividad Patrones': f"{metricas['efectividad_patrones']*100:.1f}%",
                    'Confianza Media': f"{metricas['confianza_media']:.2f}",
                    'Mejor Patrón Win Rate': f"{metricas['mejor_patron_wr']:.1f}%"
                },
                'archivos_generados': {
                    'reporte': 'reports/BTCUSDT_Backtest_' + 
                              datetime.now().strftime('%Y%m%d') + '.pdf',
                    'resultados': 'results/BTCUSDT-results.csv',
                    'balance': 'results/BTCUSDT-balance.csv',
                    'graficos': 'figures/'
                }
            }
            
        except Exception as e:
            logging.error(f"Error al preparar resumen: {str(e)}")
            return {'error': str(e), 'status': False}

def main():
    """Función principal del sistema"""
    try:
        # Iniciar sistema
        sistema = TradingSystem()
        
        # Ejecutar análisis
        resumen = sistema.ejecutar_analisis_completo()
        
        # Mostrar resultados
        if resumen['status']:
            print("\nAnálisis completado exitosamente!")
            print("\nMétricas Principales:")
            for k, v in resumen['metricas_principales'].items():
                print(f"- {k}: {v}")
                
            print("\nAnálisis de Patrones:")
            for k, v in resumen['patrones'].items():
                print(f"- {k}: {v}")
                
            print("\nArchivos Generados:")
            for k, v in resumen['archivos_generados'].items():
                print(f"- {k}: {v}")
                
        else:
            print(f"Error: {resumen['error']}")
            
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()