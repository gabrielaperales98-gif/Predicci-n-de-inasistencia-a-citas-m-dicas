# Predicción de Inasistencias médicas 
Este proyecto explora cómo el aprendizaje automático puede ayudar a identificar citas médicas con mayor probabilidad de inasistencia. El problema se relaciona con la gestión de agendas y la posibilidad de priorizar el seguimiento de las citas.

## ¿Qué hace el proyecto?

Simula 1,500 citas médicas con tres variables: días de anticipación con que se agendó la cita, número de inasistencias previas y si se registró un recordatorio. Cada cita tiene una etiqueta que indica si la persona asistió o faltó.

Entrena una regresión logística con 1,200 citas para estimar la probabilidad de inasistencia. El modelo está implementado en Python mediante descenso de gradiente.

Evalúa el modelo con las 300 citas restantes. Compara sus estimaciones con una referencia que asigna a todas las citas la misma probabilidad, calculada a partir de la tasa de inasistencia del conjunto de entrenamiento.

Genera un reporte con el puntaje de Brier, la precisión, la sensibilidad y la matriz de confusión. El umbral de clasificación se puede modificar para observar cómo cambian las inasistencias detectadas y las falsas alertas.

## Resultados

En el conjunto de prueba simulado, la regresión logística obtuvo un puntaje de Brier de 0.1706, frente a 0.1987 de la referencia. Un puntaje menor indica que las probabilidades estimadas estuvieron más cerca de los resultados observados.

Con un umbral de 0.50, el modelo detectó 19 de las 82 inasistencias y generó 5 falsas alertas. Esto muestra que, con ese umbral, identificó alrededor del 23 % de las inasistencias.

## ¿Cómo ejecutarlo?

El proyecto utiliza Python 3 y su biblioteca estándar. Desde la carpeta del repositorio, ejecuta:

python proyecto.py

El programa genera dos archivos:

citas_simuladas.csv — conjunto de datos utilizado en el proyecto.

resultados.json — métricas de la regresión logística y del método de referencia.

Para probar otro umbral, cambia el valor de UMBRAL en proyecto.py y vuelve a ejecutarlo. La semilla fija permite comparar los resultados usando las mismas citas simuladas.

## Alcance

El proyecto utiliza exclusivamente datos simulados. Los resultados reflejan los patrones definidos al generar esos datos. Para saber si el modelo sería útil en una agenda médica real, se necesitaría evaluarlo con datos reales y autorizados.
