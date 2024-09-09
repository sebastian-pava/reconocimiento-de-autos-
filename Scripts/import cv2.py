import cv2
import sys

def count_cars(image_path, car_cascade_path):
    # Cargar el clasificador en cascada
    car_cascade = cv2.CascadeClassifier(car_cascade_path)
    
    # Leer la imagen
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se puede cargar la imagen.")
        return
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detectar coches
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Dibujar rectángulos alrededor de los coches
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Contar el número de coches detectados
    car_count = len(cars)
    
    # Mostrar la imagen con coches detectados
    cv2.imshow('Detected Cars', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Número de coches detectados: {car_count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python count_cars.py <image_path> <car_cascade_path>")
        sys.exit()
    
    image_path = sys.argv[1]
    car_cascade_path = sys.argv[2]
    
    count_cars(image_path, car_cascade_path)
