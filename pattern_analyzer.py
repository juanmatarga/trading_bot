import stumpy
import numpy as np
import pandas as pd
from scipy.stats import zscore
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class PatternAnalyzer:
    def __init__(self, min_window_size=12, max_window_size=24, 
                 distance_threshold=0.35, min_neighbors=2):  # Aumentado threshold
        """
        Inicializa el analizador de patrones con parámetros ajustados.
        """
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.distance_threshold = distance_threshold
        self.min_neighbors = min_neighbors
        
        # Parámetros adicionales ajustados para mayor sensibilidad
        self.volatility_weight = 0.3  # Reducido para dar menos peso a volatilidad
        self.pattern_weight = 0.7    # Aumentado para dar más peso a patrones
        self.trend_threshold = 0.008  # Reducido significativamente
        
        logging.info(f"PatternAnalyzer inicializado con window_size={min_window_size}-"
                    f"{max_window_size}, threshold={distance_threshold}")

    def prepare_data(self, df):
        """
        Prepara los datos para el análisis con STUMPY.
        """
        try:
            if df.empty or len(df) < self.min_window_size * 2:
                return None, None
            
            # Convertir precios y manejar valores no numéricos
            prices = pd.to_numeric(df['close'], errors='coerce').fillna(method='ffill').values
            prices = prices.astype(np.float64)
            
            # Calcular retornos logarítmicos
            returns = np.diff(np.log(prices))
            
            # Normalización simple
            normalized = zscore(returns, nan_policy='omit')
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
            logging.info(f"Datos preparados exitosamente: {len(normalized)} puntos")
            return normalized, prices
            
        except Exception as e:
            logging.error(f"Error en preparación de datos: {str(e)}")
            return None, None

    def get_trading_signal(self, df, current_idx):
        """
        Obtiene señal de trading basada en patrones detectados.
        """
        try:
            normalized_data, prices = self.prepare_data(df)
            if normalized_data is None or prices is None:
                return 0, 0
            
            # Usar una ventana más pequeña para mayor sensibilidad
            window_size = self.min_window_size
            
            # Calcular matriz de perfil
            matrix_profile = stumpy.stump(normalized_data, m=window_size)
            
            # Obtener el patrón actual
            current_pattern = normalized_data[current_idx-window_size:current_idx]
            
            # Encontrar los k patrones más similares
            k = 5  # Buscar los 5 patrones más similares
            nearest_neighbors_idx = matrix_profile[:-window_size, 1]
            distances = matrix_profile[:-window_size, 0]
            
            # Ordenar por distancia
            sorted_idx = np.argsort(distances)[:k]
            similar_patterns_idx = nearest_neighbors_idx[sorted_idx]
            
            future_returns = []
            for idx in similar_patterns_idx:
                if idx + window_size + 5 >= len(prices):
                    continue
                    
                # Calcular retorno futuro después del patrón
                future_return = (prices[idx + window_size + 5] - 
                            prices[idx + window_size]) / prices[idx + window_size]
                future_returns.append(future_return)
                
                logging.info(f"Patrón similar encontrado en idx {idx}: "
                            f"retorno_futuro={future_return:.4f}")
            
            if future_returns:
                # Predecir dirección basado en patrones históricos
                avg_return = np.mean(future_returns)
                confidence = 1 - np.mean(distances[sorted_idx])
                
                # Ser menos restrictivo con los umbrales
                if abs(avg_return) > 0.005:  # 0.5% de movimiento mínimo
                    signal = 'LONG' if avg_return > 0 else 'SHORT'
                    
                    logging.info(f"Señal generada: {signal}, "
                            f"retorno_esperado={avg_return:.4f}, "
                            f"confianza={confidence:.4f}")
                    
                    return signal, confidence
            
            return 0, 0
            
        except Exception as e:
            logging.error(f"Error al obtener señal de trading: {str(e)}")
            return 0, 0

    def analyze_pattern_strength(self, normalized_data, current_idx, window_size):
        """Analiza la fuerza del patrón actual."""
        try:
            if current_idx < window_size:
                return 0, None
                
            matrix_profile = stumpy.stump(normalized_data, m=window_size)
            distances = matrix_profile[:-window_size, 0]
            
            similar_distances = distances[distances < self.distance_threshold]
            if len(similar_distances) > 0:
                pattern_strength = 1 - np.mean(similar_distances)
                return pattern_strength, None
            
            return 0, None
            
        except Exception as e:
            logging.error(f"Error al analizar fuerza del patrón: {str(e)}")
            return 0, None

    def calculate_volatility_score(self, normalized_data, window=20):
        """Calcula el score de volatilidad."""
        try:
            if len(normalized_data) < window:
                return 0
                
            recent_volatility = np.std(normalized_data[-window:])
            historical_volatility = np.std(normalized_data)
            
            if historical_volatility > 0:
                volatility_score = recent_volatility / historical_volatility
                return min(1, volatility_score)
            
            return 0
            
        except Exception as e:
            logging.error(f"Error al calcular score de volatilidad: {str(e)}")
            return 0

    def analyze_trend(self, prices, window=20):
        """Analiza la tendencia actual."""
        try:
            if len(prices) < window:
                return None, 0
                
            returns = np.diff(np.log(prices[-window:]))
            trend = np.sum(returns)
            
            # Usar umbral más bajo para tendencias
            max_strength = self.trend_threshold * window
            normalized_strength = min(1, abs(trend) / max_strength)
            
            direction = 'LONG' if trend > 0 else 'SHORT'
            return direction, normalized_strength
            
        except Exception as e:
            logging.error(f"Error al analizar tendencia: {str(e)}")
            return None, 0

    def validate_signal(self, signal, confidence, prices):
        """Valida la señal generada."""
        try:
            if signal is None:
                return None, 0
                
            # Validación menos estricta
            if confidence < 0.3:  # Reducido de 0.5 a 0.3
                return None, 0
                
            trend_direction, trend_strength = self.analyze_trend(prices)
            
            # Reducir penalización por contratendencia
            if trend_direction and signal != trend_direction:
                confidence *= (1 - trend_strength * 0.5)  # Penalización reducida
            
            return signal, confidence
            
        except Exception as e:
            logging.error(f"Error en validación de señal: {str(e)}")
            return None, 0

    def calculate_risk_parameters(self, prices, signal, confidence):
        """Calcula parámetros de riesgo para la operación."""
        try:
            if signal is None:
                return None
                
            current_price = prices[-1]
            volatility = np.std(np.diff(np.log(prices[-20:])))
            
            # Ajustar multiplicadores para tomar más trades
            base_tp = volatility * 1.5  # Reducido de 2 a 1.5
            base_sl = volatility * 1.0  # Reducido de 1.5 a 1.0
            
            tp_multiplier = 1 + confidence * 0.5  # Reducido el efecto de la confianza
            sl_multiplier = 1 - confidence * 0.3  # Reducido el efecto de la confianza
            
            take_profit = current_price * (1 + (base_tp * tp_multiplier))
            stop_loss = current_price * (1 - (base_sl * sl_multiplier))
            
            if signal == 'SHORT':
                take_profit, stop_loss = stop_loss, take_profit
                
            return {
                'entry_price': current_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'volatility': volatility
            }
            
        except Exception as e:
            logging.error(f"Error al calcular parámetros de riesgo: {str(e)}")
            return None

    def visualize_pattern(self, df, current_idx, window_size):
        """Visualiza el patrón actual y sus similares."""
        try:
            if not os.path.exists('figures/patterns'):
                os.makedirs('figures/patterns')
                
            normalized_data, prices = self.prepare_data(df)
            if normalized_data is None:
                return
                
            matrix_profile = stumpy.stump(normalized_data, m=window_size)
            
            plt.figure(figsize=(15, 8))
            current_pattern = normalized_data[current_idx-window_size:current_idx]
            plt.plot(range(window_size), current_pattern, 'b-', 
                    label='Patrón Actual', linewidth=2)
            
            nearest_neighbors = matrix_profile[:-window_size, 1]
            distances = matrix_profile[:-window_size, 0]
            
            for idx, dist in zip(nearest_neighbors, distances):
                if dist < self.distance_threshold:
                    pattern = normalized_data[idx:idx+window_size]
                    plt.plot(range(window_size), pattern, 'g-', alpha=0.3)
            
            plt.title('Patrón Actual vs Patrones Similares')
            plt.xlabel('Tiempo')
            plt.ylabel('Retorno Normalizado')
            plt.legend()
            plt.grid(True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'figures/patterns/pattern_{timestamp}.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al visualizar patrón: {str(e)}")

def main():
    """Función principal para pruebas"""
    try:
        from binance_data import BinanceDataLoader
        
        loader = BinanceDataLoader()
        df = loader.obtener_datos_historicos()
        
        if df.empty:
            logging.error("No hay datos disponibles")
            return
            
        analyzer = PatternAnalyzer()
        current_idx = len(df) - 1
        signal, confidence = analyzer.get_trading_signal(df, current_idx)
        
        print(f"Señal: {signal}, Confianza: {confidence:.2f}")
        
        if signal:
            analyzer.visualize_pattern(df, current_idx, analyzer.min_window_size)
        
    except Exception as e:
        logging.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()