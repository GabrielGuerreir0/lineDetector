import cv2
import numpy as np
from Detector.detector import UltrafastLaneDetector, ModelType

# Configurações
model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False
image_path = "G0032817.jpg"

usar_coordenadas_hardcode = False

manual_coordinates = [
    ((100, 200), (150, 600)),  # Faixa 1
    ((552, 159), (692, 169)),  # Faixa 2
    ((54, 439), (239, 509)),  # Faixa 3
    ((700, 200), (750, 600)),  # Faixa 4
]

# === Leitura da imagem ===
try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")
except Exception as e:
    print(f"Erro ao carregar imagem: {e}")
    exit()

original_height, original_width = img.shape[:2]
print(f"[ORIGINAL] Resolução da imagem: {original_width}x{original_height}")

# === Inicializa o detector de faixas ===
try:
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
except Exception as e:
    print(f"Erro ao inicializar detector: {e}")
    exit()

detected_coordinates = []

# === Detecção de faixas ===
try:
    output_img, lanes_data = lane_detector.detect_lanes(img)
    output_height, output_width = output_img.shape[:2]
    print(f"[DETECÇÃO] Resolução da imagem com faixas detectadas: {output_width}x{output_height}")

    first_and_last_coords = lane_detector.get_first_and_last_coordinates()

    for i, (first, last) in enumerate(first_and_last_coords):
        print(f"Faixa {i+1}: Primeira coordenada: {first}, Última coordenada: {last}")
        detected_coordinates.append((first, last))

    cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected lanes", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if cv2.imwrite("output.jpg", output_img):
        print("Resultado salvo com sucesso em output.jpg")
    else:
        print("Erro ao salvar resultado em output.jpg")

except Exception as e:
    print(f"Erro durante processamento: {e}")
    exit()

# === Usa coordenadas definidas manualmente (se ativado) ===
if usar_coordenadas_hardcode:
    detected_coordinates = manual_coordinates.copy()
    print("\nUsando coordenadas definidas em hard-code:")
else:
    print("\nCoordenadas detectadas automaticamente:")

for i, (first, last) in enumerate(detected_coordinates):
    print(f"Faixa {i+1}: Primeira coordenada: {first}, Última coordenada: {last}")

if len(detected_coordinates) < 3:
    print("Erro: menos de 3 faixas disponíveis em detected_coordinates.")
    exit()

# === Ordena pontos para transformação de perspectiva ===
# Pegando a faixa 2 (direita) e faixa 3 (esquerda)
p2_top, p2_bottom = sorted(detected_coordinates[1], key=lambda p: p[1])
p3_top, p3_bottom = sorted(detected_coordinates[2], key=lambda p: p[1])

print(f"[PONTOS] Faixa 2: {p2_top}, {p2_bottom}")
print(f"[PONTOS] Faixa 3: {p3_top}, {p3_bottom}")
 
te= p2_top
td= p3_top
be= p2_bottom
bd= p3_bottom

# Pontos de origem para transformação (ordem: top-left, top-right, bottom-left, bottom-right)
pts1 = np.float32([

  te ,be,      
  td,  bd  
])

# === Desenha os pontos e linhas na imagem ===
img_with_points = output_img.copy()
for i, point in enumerate(pts1):
    cv2.circle(img_with_points, tuple(map(int, point)), 8, (0, 255, 0), -1)
    cv2.putText(img_with_points, f"P{i+1}", tuple(map(int, point + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Desenha linhas entre os pontos (para verificar se o retângulo está correto)
for i in range(4):
    pt1 = tuple(map(int, pts1[i]))
    pt2 = tuple(map(int, pts1[(i + 1) % 4]))
    cv2.line(img_with_points, pt1, pt2, (255, 0, 0), 2)

print(f"[PONTOS] Resolução da imagem com pontos desenhados: {img_with_points.shape[1]}x{img_with_points.shape[0]}")

cv2.namedWindow("Pontos na Imagem Original", cv2.WINDOW_NORMAL)
cv2.imshow("Pontos na Imagem Original", img_with_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

if cv2.imwrite("imagem_com_pontos.jpg", img_with_points):
    print("Imagem com pontos marcada salva como imagem_com_pontos.jpg")
else:
    print("Erro ao salvar imagem_com_pontos.jpg")

# === Transformação de perspectiva ===
pts2 = np.float32([
    (0, 0),
    (0, 720),
    (1200, 0),
    (1200, 720)
])

image =cv2.imread(image_path)
frame = cv2.resize(image, (1200, 720))
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warped_img = cv2.warpPerspective(frame, matrix, (1200, 720))

print(f"[WARP] Resolução da imagem após transformação de perspectiva: {warped_img.shape[1]}x{warped_img.shape[0]}")

cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
cv2.imshow("Warped Image", warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

if cv2.imwrite("warped_output.jpg", warped_img):
    print("Imagem transformada salva com sucesso em warped_output.jpg")
else:
    print("Erro ao salvar warped_output.jpg")
