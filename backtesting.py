import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance_data import BinanceDataLoader
from pattern_analyzer import PatternAnalyzer
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class Backtester:
    def __init__(self):
        """
        Inicializa el backtester con parámetros ajustados para mayor sensibilidad.
        """
        try:
            logging.info("Iniciando Backtester")
            
            # Parámetros de trading
            self.symbol = 'BTCUSDT'
            self.capital_inicial = 2500
            self.apalancamiento = 10
            self.comision = 0.0004  # 0.04%
            
            # Parámetros de gestión de riesgo ajustados
            self.risk_per_trade = 0.02  # 2% del capital por operación
            self.max_positions = 3
            self.min_confidence = 0.5  # Reducido de 0.6 a 0.5 para más señales
            
            # Parámetros de take profit y stop loss más accesibles
            self.tp_multiplier = 1.5  # Reducido de 2 a 1.5
            self.sl_multiplier = 1.0  # Reducido de 1.5 a 1.0
            
            # Crear directorios necesarios
            os.makedirs('data', exist_ok=True)
            os.makedirs('results', exist_ok=True)
            
            # Inicializar componentes
            self.data_loader = BinanceDataLoader()
            self.pattern_analyzer = PatternAnalyzer(
                min_window_size=12,  # Reducido de 24 a 12 (2 días)
                max_window_size=36,  # Reducido de 48 a 36 (6 días)
                distance_threshold=0.15,  # Aumentado de 0.1 a 0.15
                min_neighbors=2  # Reducido de 3 a 2
            )
            
            self.posiciones_abiertas = []
            logging.info("Backtester inicializado con parámetros ajustados")
            
        except Exception as e:
            logging.error(f"Error en inicialización del Backtester: {str(e)}")
            raise
            
    def ejecutar_backtest(self):
        """
        Ejecuta el proceso de backtesting con validaciones mejoradas.
        
        Returns:
            tuple: (resultados, balance, fechas_balance)
        """
        try:
            logging.info(f"Iniciando proceso de backtesting para {self.symbol}")
            
            # Obtener datos históricos
            df = self.data_loader.obtener_datos_historicos()
            if df.empty:
                raise ValueError("No hay datos disponibles para el backtest")
                
            # Validar datos
            self._validar_datos(df)
            
            # Inicializar resultados
            resultados = []
            balance = [self.capital_inicial]
            fechas_balance = [df.index[0]]
            
            # Ejecutar backtest
            for idx in range(len(df)):
                try:
                    # Actualizar posiciones abiertas
                    self._actualizar_posiciones(df, idx, resultados, balance)
                    
                    # Verificar si podemos abrir nuevas posiciones
                    if len(self.posiciones_abiertas) < self.max_positions:
                        # Obtener señal de trading
                        signal, confidence = self.pattern_analyzer.get_trading_signal(
                            df.iloc[:idx+1], idx
                        )
                        
                        # Registrar señal para análisis
                        self._registrar_senal(df.index[idx], signal, confidence)
                        
                        # Abrir posición si hay señal válida
                        if abs(signal) > 0 and confidence >= self.min_confidence:
                            self._abrir_posicion(
                                df, idx, signal, confidence, balance[-1]
                            )
                    
                    # Actualizar balance
                    fechas_balance.append(df.index[idx])
                    balance.append(balance[-1])
                    
                except Exception as e:
                    logging.error(f"Error en iteración {idx}: {str(e)}")
                    continue
                
            # Cerrar posiciones abiertas al final
            self._cerrar_posiciones_pendientes(df, len(df)-1, resultados, balance)
            
            # Guardar resultados
            self._guardar_resultados(resultados, balance, fechas_balance)
            
            logging.info(f"Backtest completado: {len(resultados)} operaciones")
            return resultados, balance, fechas_balance
            
        except Exception as e:
            logging.error(f"Error en ejecución de backtest: {str(e)}")
            return [], [], []
            
    def _validar_datos(self, df):
        """
        Valida los datos de entrada del backtest.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos
        """
        try:
            # Verificar columnas requeridas
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Faltan columnas requeridas en los datos")
                
            # Verificar cantidad mínima de datos
            min_data = self.pattern_analyzer.min_window_size * 3
            if len(df) < min_data:
                raise ValueError(
                    f"Insuficientes datos. Se requieren al menos {min_data} registros"
                )
                
            # Verificar valores nulos
            if df[required_columns].isnull().any().any():
                raise ValueError("Los datos contienen valores nulos")
                
            logging.info(f"Datos validados: {len(df)} registros")
            
        except Exception as e:
            logging.error(f"Error en validación de datos: {str(e)}")
            raise
            
    def _actualizar_posiciones(self, df, idx, resultados, balance):
        """
        Actualiza el estado de las posiciones abiertas.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos
            idx (int): Índice actual
            resultados (list): Lista de resultados
            balance (list): Lista de balance
        """
        try:
            precio_actual = df['close'].iloc[idx]
            
            for posicion in self.posiciones_abiertas[:]:  # Copiar lista para iterar
                # Verificar take profit
                if (posicion['direccion'] == 'LONG' and 
                    precio_actual >= posicion['take_profit']) or \
                   (posicion['direccion'] == 'SHORT' and 
                    precio_actual <= posicion['take_profit']):
                    self._cerrar_posicion(
                        posicion, idx, df.index[idx], precio_actual, 
                        'TP', resultados, balance
                    )
                    continue
                    
                # Verificar stop loss
                if (posicion['direccion'] == 'LONG' and 
                    precio_actual <= posicion['stop_loss']) or \
                   (posicion['direccion'] == 'SHORT' and 
                    precio_actual >= posicion['stop_loss']):
                    self._cerrar_posicion(
                        posicion, idx, df.index[idx], precio_actual, 
                        'SL', resultados, balance
                    )
                    
        except Exception as e:
            logging.error(f"Error al actualizar posiciones: {str(e)}")
            
    def _abrir_posicion(self, df, idx, signal, confidence, balance_actual):
        """
        Abre una nueva posición de trading.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos
            idx (int): Índice actual
            signal (int): Señal de trading (-1, 1)
            confidence (float): Nivel de confianza
            balance_actual (float): Balance actual
        """
        try:
            precio_entrada = df['close'].iloc[idx]
            volatilidad = self._calcular_volatilidad(df, idx)
            
            # Calcular tamaño de la posición basado en gestión de riesgo
            riesgo_capital = balance_actual * self.risk_per_trade
            stop_loss_pips = volatilidad * self.sl_multiplier
            
            tamanio_posicion = (riesgo_capital / stop_loss_pips) * self.apalancamiento
            
            if signal > 0:  # LONG
                stop_loss = precio_entrada * (1 - self.sl_multiplier * volatilidad)
                take_profit = precio_entrada * (1 + self.tp_multiplier * volatilidad)
                direccion = 'LONG'
            else:  # SHORT
                stop_loss = precio_entrada * (1 + self.sl_multiplier * volatilidad)
                take_profit = precio_entrada * (1 - self.tp_multiplier * volatilidad)
                direccion = 'SHORT'
                
            posicion = {
                'entrada_idx': idx,
                'fecha_entrada': df.index[idx],
                'direccion': direccion,
                'precio_entrada': precio_entrada,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'tamanio': tamanio_posicion,
                'confianza': confidence
            }
            
            self.posiciones_abiertas.append(posicion)
            logging.info(
                f"Nueva posición {direccion} abierta a {precio_entrada:.2f} "
                f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f})"
            )
            
        except Exception as e:
            logging.error(f"Error al abrir posición: {str(e)}")
            
    def _calcular_volatilidad(self, df, idx):
        """
        Calcula la volatilidad actual del mercado.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos
            idx (int): Índice actual
            
        Returns:
            float: Volatilidad calculada
        """
        try:
            # Usar últimas 24 velas (4 días con velas de 4 horas)
            window = 24
            start_idx = max(0, idx - window)
            
            precios = df['close'].iloc[start_idx:idx+1]
            returns = np.log(precios / precios.shift(1)).dropna()
            
            return returns.std()
            
        except Exception as e:
            logging.error(f"Error al calcular volatilidad: {str(e)}")
            return 0.02  # Valor por defecto
            
    def _cerrar_posicion(self, posicion, idx, fecha, precio_cierre, 
                        resultado, resultados, balance):
        """
        Cierra una posición existente.
        
        Args:
            posicion (dict): Diccionario con datos de la posición
            idx (int): Índice actual
            fecha (datetime): Fecha de cierre
            precio_cierre (float): Precio de cierre
            resultado (str): Resultado de la operación ('TP' o 'SL')
            resultados (list): Lista de resultados
            balance (list): Lista de balance
        """
        try:
            # Calcular retorno
            if posicion['direccion'] == 'LONG':
                retorno = (precio_cierre - posicion['precio_entrada']) / posicion['precio_entrada']
            else:
                retorno = (posicion['precio_entrada'] - precio_cierre) / posicion['precio_entrada']
                
            # Aplicar apalancamiento y comisiones
            retorno = (retorno * self.apalancamiento) - (self.comision * 2)
            
            # Actualizar balance
            profit_loss = balance[-1] * retorno * self.risk_per_trade
            nuevo_balance = balance[-1] + profit_loss
            balance[-1] = nuevo_balance
            
            # Registrar operación
            operacion = {
                'fecha_entrada': posicion['fecha_entrada'],
                'fecha_salida': fecha,
                'direccion': posicion['direccion'],
                'precio_entrada': posicion['precio_entrada'],
                'precio_salida': precio_cierre,
                'stop_loss': posicion['stop_loss'],
                'take_profit': posicion['take_profit'],
                'resultado': resultado,
                'retorno_operacion': retorno,
                'confianza_patron': posicion['confianza']
            }
            
            resultados.append(operacion)
            self.posiciones_abiertas.remove(posicion)
            
            logging.info(
                f"Posición {posicion['direccion']} cerrada: {resultado} "
                f"(Retorno: {retorno*100:.2f}%)"
            )
            
        except Exception as e:
            logging.error(f"Error al cerrar posición: {str(e)}")
            
    def _cerrar_posiciones_pendientes(self, df, idx, resultados, balance):
        """
        Cierra todas las posiciones pendientes al final del backtest.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos
            idx (int): Índice final
            resultados (list): Lista de resultados
            balance (list): Lista de balance
        """
        try:
            precio_final = df['close'].iloc[idx]
            fecha_final = df.index[idx]
            
            for posicion in self.posiciones_abiertas[:]:
                self._cerrar_posicion(
                    posicion, idx, fecha_final, precio_final, 
                    'CLOSE', resultados, balance
                )
                
        except Exception as e:
            logging.error(f"Error al cerrar posiciones pendientes: {str(e)}")
            
    def _registrar_senal(self, fecha, signal, confidence):
        """
        Registra las señales generadas para análisis posterior.
        
        Args:
            fecha (datetime): Fecha de la señal
            signal (int): Señal generada
            confidence (float): Nivel de confianza
        """
        try:
            with open('results/signals.log', 'a') as f:
                f.write(f"{fecha},{signal},{confidence:.4f}\n")
                
        except Exception as e:
            logging.error(f"Error al registrar señal: {str(e)}")
            
    def _guardar_resultados(self, resultados, balance, fechas_balance):
        """
        Guarda los resultados del backtest.
        
        Args:
            resultados (list): Lista de operaciones
            balance (list): Lista de balance
            fechas_balance (list): Lista de fechas
        """
        try:
            # Guardar operaciones
            df_ops = pd.DataFrame(resultados)
            df_ops.to_csv(f'results/{self.symbol}-results.csv', index=False)
            
            # Guardar balance
            df_balance = pd.DataFrame({
                'fecha': fechas_balance,
                'balance': balance
            })
            df_balance.to_csv(f'results/{self.symbol}-balance.csv', index=False)
            
            logging.info("Resultados guardados exitosamente")
            
        except Exception as e:
            logging.error(f"Error al guardar resultados: {str(e)}")
            raise

def main():
    """Función principal para pruebas"""
    try:
        backtester = Backtester()
        resultados, balance, fechas = backtester.ejecutar_backtest()
        
        print(f"Total operaciones: {len(resultados)}")
        if balance:
            print(f"Balance final: ${balance[-1]:.2f}")
            print(f"Retorno total: {((balance[-1]/balance[0])-1)*100:.2f}%")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()