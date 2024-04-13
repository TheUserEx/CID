# Hands On 2 - Hernandez De la Cruz Miguel Angel

class RegresionLinealSimple:  # Defino la clase
    def __init__(self):  # Método constructor
        self.b0 = None  # Inicializa el coeficiente b0 como nulo
        self.b1 = None

                  #Parametros del método SELF
    def fit(self, X, y):  # Método para ajustar el modelo
        n = len(X)  # Longitud de la lista para X
        
        # Calculando las sumas necesarias
        sum_x = 0  # Inicializa la suma de los datos de entrada (SumX)
        sum_y = 0  #                                            (SumY)
        sum_xy = 0  #                                           (SumXY)
        sum_x_squared = 0  #                                    (SumXquad)
        sum_x_times_sum_y = 0  #                                (SumX*SumY)
        sum_x_sum_x = 0  #                                      (SumXSumX)
        n_times_sum_xy = 0  #                                   (n*SumXY)
        sum_x_sum_y = 0  #                                      (SumXSumY)
        n_times_sum_xy_minus_sum_x_sum_y = 0  #                 (n*SumXY-SumX*SumY)
        
        for i in range(n):  # Itera sobre los datos de entrada
            sum_x += X[i]  # Agrega el dato a la suma correspondiente
            sum_y += y[i]  
            sum_xy += X[i] * y[i]  
            sum_x_squared += X[i] ** 2  
            sum_x_times_sum_y += X[i] * sum_y  
            sum_x_sum_x += X[i] * sum_x  
            n_times_sum_xy += n * sum_xy  
            sum_x_sum_y += sum_x * sum_y  
            n_times_sum_xy_minus_sum_x_sum_y += n * sum_xy - sum_x * sum_y  

        # Se calculan los coeficientes B0 y B1
        self.b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        self.b0 = (sum_y - self.b1 * sum_x) / n

    def predict(self, x):  # Método/predicciones
        if self.b0 is None or self.b1 is None:  # Si los coeficientes no han sido asignados
            raise Exception("Error, primero ajuste el modelo antes de predecir.") 
        return self.b0 + self.b1 * x  # Retorna la predicción de los coeficientes calculados


# Datos
X = [1, 2, 3, 4, 5, 6, 7, 8, 9]                     # Advertising
y = [2, 4, 6, 8, 10, 12, 14, 16, 18]                # Sales

# Crear y entrenar el modelo (Implica estimar los parámetros del modelo)
model = RegresionLinealSimple()  # Instancia un objeto de la clase
model.fit(X, y)  # Ajusta el modelo a los datos de entrada y salida



# Imprimir los valores de B0 y B1


print("\n\n\n_________________________________________________________________________________________________\n")
print("                            Hands-on 2: Coding, SLR Technique")
print("_________________________________________________________________________________________________\n\n\n")

print("                                  B0:", model.b0) 
print("                                  B1:", model.b1)  
print("\n\n\n_________________________________________________________________________________________________\n")



# Hacer predicciones
nuevoX = 20  # Nuevo dato de entrada
prediccionDeY = model.predict(nuevoX)  # Realiza una predicción basada en el nuevo dato de entrada

print("             Para Advertising =", nuevoX, ", la predicción de Sales es:", prediccionDeY)  # Imprime la predicción realizada
print("_________________________________________________________________________________________________\n\n\n")





