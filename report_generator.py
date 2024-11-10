from fpdf import FPDF
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class TradingReport(FPDF):
    def header(self):
        """Define el encabezado de cada página del reporte"""
        # Logo (si existe)
        if os.path.exists('logo.png'):
            self.image('logo.png', 10, 8, 33)
            
        # Título del reporte
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Reporte de Backtesting - BTCUSDT', 0, 1, 'C')
        
        # Fecha del reporte
        self.set_font('Arial', 'I', 8)
        self.cell(0, 5, f'Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
        
        # Línea separadora
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        """Define el pie de página"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

class ReportGenerator:
    def __init__(self):
        """Inicializa el generador de reportes"""
        self.pdf = TradingReport()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        os.makedirs('reports', exist_ok=True)
        logging.info("ReportGenerator inicializado")
        
    def generar_reporte(self, metricas, ratios, drawdown, df_ops):
        """
        Genera un reporte completo de backtesting.
        
        Args:
            metricas (dict): Métricas de rendimiento
            ratios (dict): Ratios financieros
            drawdown (dict): Métricas de drawdown
            df_ops (pandas.DataFrame): DataFrame de operaciones
        """
        try:
            if df_ops.empty:
                raise ValueError("No hay datos suficientes para generar el reporte")
                
            logging.info("Iniciando generación del reporte")
            
            # Agregar secciones del reporte
            self.agregar_resumen_ejecutivo(metricas, ratios, drawdown)
            self.agregar_analisis_patrones(metricas, df_ops)
            self.agregar_metricas_detalladas(metricas)
            self.agregar_analisis_riesgo(ratios, drawdown)
            self.agregar_graficos()
            self.agregar_tabla_operaciones(df_ops)
            self.agregar_conclusiones(metricas, ratios, drawdown)
            
            self.guardar_reporte()
            logging.info("Reporte generado exitosamente")
            
        except Exception as e:
            logging.error(f"Error al generar reporte: {str(e)}")
            raise
            
    def agregar_resumen_ejecutivo(self, metricas, ratios, drawdown):
        """Agrega resumen ejecutivo al reporte"""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Arial', 'B', 14)
            self.pdf.cell(0, 10, 'Resumen Ejecutivo', 0, 1)
            self.pdf.ln(5)
            
            self.pdf.set_font('Arial', '', 10)
            
            # Métricas principales
            metricas_resumen = [
                ('Total Operaciones', f"{metricas['total_operaciones']}"),
                ('Win Rate Total', f"{metricas['win_rate_total']:.1f}%"),
                ('Profit Factor', f"{ratios['profit_factor']:.2f}"),
                ('Sharpe Ratio', f"{ratios['sharpe_ratio']:.2f}"),
                ('Máximo Drawdown', f"{drawdown['max_drawdown']*100:.1f}%"),
                ('Retorno Total', f"{drawdown['retorno_total']:.1f}%")
            ]
            
            for titulo, valor in metricas_resumen:
                self.pdf.cell(60, 8, titulo, 0)
                self.pdf.cell(0, 8, valor, 0, 1)
                
            self.pdf.ln(5)
            
        except Exception as e:
            logging.error(f"Error al agregar resumen ejecutivo: {str(e)}")
            
    def agregar_analisis_patrones(self, metricas, df_ops):
        """Agrega análisis de patrones al reporte"""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Arial', 'B', 14)
            self.pdf.cell(0, 10, 'Análisis de Patrones', 0, 1)
            self.pdf.ln(5)
            
            self.pdf.set_font('Arial', '', 10)
            
            # Métricas de patrones
            self.pdf.cell(0, 8, 'Efectividad de Patrones:', 0, 1)
            self.pdf.ln(2)
            
            metricas_patrones = [
                ('Confianza Media', f"{metricas['confianza_media_patrones']:.2f}"),
                ('Efectividad', f"{metricas['efectividad_patrones']*100:.1f}%"),
                ('Win Rate Alta Confianza', self._calcular_win_rate_alta_confianza(df_ops))
            ]
            
            for titulo, valor in metricas_patrones:
                self.pdf.cell(60, 8, titulo, 0)
                self.pdf.cell(0, 8, valor, 0, 1)
                
            # Agregar gráficos de patrones si existen
            self._agregar_graficos_patrones()
            
        except Exception as e:
            logging.error(f"Error al agregar análisis de patrones: {str(e)}")
            
    def _calcular_win_rate_alta_confianza(self, df_ops):
        """Calcula el win rate para operaciones de alta confianza"""
        try:
            ops_alta_confianza = df_ops[df_ops['confianza_patron'] >= 0.7]
            if len(ops_alta_confianza) > 0:
                wins = len(ops_alta_confianza[ops_alta_confianza['resultado'] == 'TP'])
                return f"{(wins/len(ops_alta_confianza))*100:.1f}%"
            return "N/A"
            
        except Exception as e:
            logging.error(f"Error al calcular win rate alta confianza: {str(e)}")
            return "N/A"
            
    def _agregar_graficos_patrones(self):
        """Agrega visualizaciones de patrones al reporte"""
        try:
            pattern_dir = 'figures/patterns'
            if os.path.exists(pattern_dir):
                pattern_files = [f for f in os.listdir(pattern_dir) 
                               if f.endswith('.png')]
                
                if pattern_files:
                    self.pdf.add_page()
                    self.pdf.set_font('Arial', 'B', 12)
                    self.pdf.cell(0, 10, 'Ejemplos de Patrones Detectados', 0, 1)
                    
                    for i, file in enumerate(sorted(pattern_files)[-3:]):  # últimos 3 patrones
                        self.pdf.image(
                            f'{pattern_dir}/{file}',
                            x=10,
                            y=40 + i*70,
                            w=190
                        )
                        
        except Exception as e:
            logging.error(f"Error al agregar gráficos de patrones: {str(e)}")
            
    def agregar_metricas_detalladas(self, metricas):
        """Agrega métricas detalladas al reporte"""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Arial', 'B', 14)
            self.pdf.cell(0, 10, 'Métricas Detalladas', 0, 1)
            self.pdf.ln(5)
            
            self.pdf.set_font('Arial', '', 10)
            
            # Métricas por dirección
            self.pdf.cell(0, 8, 'Rendimiento por Dirección:', 0, 1)
            self.pdf.ln(2)
            
            metricas_direccion = [
                ('Win Rate LONG', f"{metricas['win_rate_long']:.1f}%"),
                ('Win Rate SHORT', f"{metricas['win_rate_short']:.1f}%")
            ]
            
            for titulo, valor in metricas_direccion:
                self.pdf.cell(60, 8, titulo, 0)
                self.pdf.cell(0, 8, valor, 0, 1)
                
            self.pdf.ln(5)
            
            # Métricas de retorno
            self.pdf.cell(0, 8, 'Métricas de Retorno:', 0, 1)
            self.pdf.ln(2)
            
            metricas_retorno = [
                ('Retorno Medio', f"{metricas['retorno_medio']*100:.2f}%"),
                ('Desviación Estándar', f"{metricas['retorno_std']*100:.2f}%")
            ]
            
            for titulo, valor in metricas_retorno:
                self.pdf.cell(60, 8, titulo, 0)
                self.pdf.cell(0, 8, valor, 0, 1)
                
        except Exception as e:
            logging.error(f"Error al agregar métricas detalladas: {str(e)}")
            
    def agregar_analisis_riesgo(self, ratios, drawdown):
        """Agrega análisis de riesgo al reporte"""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Arial', 'B', 14)
            self.pdf.cell(0, 10, 'Análisis de Riesgo', 0, 1)
            self.pdf.ln(5)
            
            self.pdf.set_font('Arial', '', 10)
            
            # Ratios de riesgo
            ratios_riesgo = [
                ('Sharpe Ratio', f"{ratios['sharpe_ratio']:.2f}"),
                ('Sortino Ratio', f"{ratios['sortino_ratio']:.2f}"),
                ('Calmar Ratio', f"{ratios['calmar_ratio']:.2f}"),
                ('Máximo Drawdown', f"{drawdown['max_drawdown']*100:.1f}%"),
                ('Duración Drawdown', str(drawdown['drawdown_duration']))
            ]
            
            for titulo, valor in ratios_riesgo:
                self.pdf.cell(60, 8, titulo, 0)
                self.pdf.cell(0, 8, valor, 0, 1)
                
        except Exception as e:
            logging.error(f"Error al agregar análisis de riesgo: {str(e)}")
            
    def agregar_graficos(self):
        """Agrega gráficos al reporte"""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Arial', 'B', 14)
            self.pdf.cell(0, 10, 'Gráficos de Análisis', 0, 1)
            
            # Agregar gráficos si existen
            for grafico in ['balance.png', 'drawdown.png', 'returns.png']:
                if os.path.exists(f'figures/{grafico}'):
                    self.pdf.image(
                        f'figures/{grafico}',
                        x=10,
                        y=self.pdf.get_y(),
                        w=190
                    )
                    self.pdf.ln(100)
                    
        except Exception as e:
            logging.error(f"Error al agregar gráficos: {str(e)}")
            
    def agregar_tabla_operaciones(self, df_ops):
        """Agrega tabla de operaciones al reporte"""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Arial', 'B', 14)
            self.pdf.cell(0, 10, 'Resumen de Operaciones', 0, 1)
            self.pdf.ln(5)
            
            # Mostrar últimas 20 operaciones
            df_ultimas = df_ops.tail(20).copy()
            df_ultimas['retorno_operacion'] = df_ultimas['retorno_operacion'].map(
                lambda x: f"{x*100:.2f}%"
            )
            df_ultimas['confianza_patron'] = df_ultimas['confianza_patron'].map(
                lambda x: f"{x:.2f}"
            )
            
            # Configurar tabla
            self.pdf.set_font('Arial', 'B', 8)
            cols = ['Fecha', 'Dirección', 'Resultado', 'Retorno', 'Confianza']
            col_widths = [35, 25, 25, 25, 25]
            
            # Encabezados
            for i, col in enumerate(cols):
                self.pdf.cell(col_widths[i], 7, col, 1)
            self.pdf.ln()
            
            # Datos
            self.pdf.set_font('Arial', '', 8)
            for _, row in df_ultimas.iterrows():
                self.pdf.cell(35, 6, row['fecha_entrada'].split()[0], 1)
                self.pdf.cell(25, 6, row['direccion'], 1)
                self.pdf.cell(25, 6, row['resultado'], 1)
                self.pdf.cell(25, 6, row['retorno_operacion'], 1)
                self.pdf.cell(25, 6, row['confianza_patron'], 1)
                self.pdf.ln()
                
        except Exception as e:
            logging.error(f"Error al agregar tabla de operaciones: {str(e)}")
            
    def agregar_conclusiones(self, metricas, ratios, drawdown):
        """Agrega conclusiones al reporte"""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Arial', 'B', 14)
            self.pdf.cell(0, 10, 'Conclusiones', 0, 1)
            self.pdf.ln(5)
            
            self.pdf.set_font('Arial', '', 10)
            conclusiones = self._generar_conclusiones(metricas, ratios, drawdown)
            
            for conclusion in conclusiones.split('\n\n'):
                self.pdf.multi_cell(0, 5, conclusion)
                self.pdf.ln(5)
                
        except Exception as e:
            logging.error(f"Error al agregar conclusiones: {str(e)}")
            
    def _generar_conclusiones(self, metricas, ratios, drawdown):
        """Genera conclusiones basadas en los resultados"""
        try:
            conclusiones = []
            
            # Análisis general
            if metricas['win_rate_total'] > 50:
                conclusiones.append(
                    f"La estrategia muestra un win rate positivo de "
                    f"{metricas['win_rate_total']:.1f}%, superando el mínimo "
                    f"requerido del 50%."
                )
            else:
                conclusiones.append(
                    f"El win rate de {metricas['win_rate_total']:.1f}% está por "
                    f"debajo del mínimo requerido (50%)."
                )
                
            # Análisis de patrones
            if metricas['efectividad_patrones'] > 0.6:
                conclusiones.append(
                    f"Los patrones detectados muestran una efectividad alta "
                    f"({metricas['efectividad_patrones']*100:.1f}%), validando "
                    f"la estrategia de detección."
                )
            else:
                conclusiones.append(
                    f"La efectividad de los patrones "
                    f"({metricas['efectividad_patrones']*100:.1f}%) necesita "
                    f"optimización."
                )
                
            # Análisis de riesgo
            if ratios['sharpe_ratio'] > 1:
                conclusiones.append(
                    f"La estrategia muestra una buena relación riesgo/rendimiento "
                    f"con un Sharpe Ratio de {ratios['sharpe_ratio']:.2f}."
                )
            else:
                conclusiones.append(
                    "La relación riesgo/rendimiento necesita optimización."
                )
                
            if abs(drawdown['max_drawdown']) > 0.2:
                conclusiones.append(
                    f"El drawdown máximo de {drawdown['max_drawdown']*100:.1f}% "
                    f"es significativo. Considerar ajustar el manejo de riesgo."
                )
                
            return "\n\n".join(conclusiones)
            
        except Exception as e:
            logging.error(f"Error al generar conclusiones: {str(e)}")
            return "Error al generar conclusiones."
            
    def guardar_reporte(self):
        """Guarda el reporte en un archivo PDF"""
        try:
            filename = f"reports/BTCUSDT_Backtest_{datetime.now().strftime('%Y%m%d')}.pdf"
            self.pdf.output(filename)
            logging.info(f"Reporte guardado exitosamente: {filename}")
            
        except Exception as e:
            logging.error(f"Error al guardar el reporte: {str(e)}")
            raise

def main():
    """Función principal para pruebas"""
    try:
        from analysis import BacktestAnalyzer
        
        analyzer = BacktestAnalyzer()
        df_ops, df_balance = analyzer.cargar_resultados()
        
        if df_ops.empty or df_balance.empty:
            logging.warning("No hay datos suficientes para generar el reporte")
            return
            
        metricas = analyzer.calcular_metricas_rendimiento(df_ops)
        ratios = analyzer.calcular_ratios(df_ops, df_balance)
        drawdown = analyzer.calcular_drawdown(df_balance)
        
        report_gen = ReportGenerator()
        report_gen.generar_reporte(metricas, ratios, drawdown, df_ops)
        
        print("Reporte generado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()