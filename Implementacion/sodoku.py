print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
import sudukoSolver

########################################################################
pathImage = "Recursos/2.jpg"
heightImg = 450
widthImg = 450
model = intializePredectionModel()  # LOAD THE CNN MODEL
########################################################################


#### 1. PREPARACION DE LA IMAGEN
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # CAMBIAR EL TAMAÑO DE LA IMAGEN PARA HACERLA UNA IMAGEN CUADRADA
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREAR UNA IMAGEN EN BLANCO PARA PROBAR SI FUNCIONA LOS TAMAÑOS INDICADOS
imgThreshold = preProcess(img)

#### 2. ENCONTRAR LOS CONTORNOS
imgContours = img.copy() # HACEMOS UNA COPIA DE LA IMAGEN PARA VISUALIZARLA
imgBigContour = img.copy() # COPIA DEL CONTORNO DE LA IMAGEN ENTERA
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # BUSCAR TODOS LOS CONTONOS FUERA DE LOS MARGENES DEL SUDOKU
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DIBUJA TODOS LOS CONTORNOS EXTERNOS

#### 3. ENCONTRAR  LOS MARGENES DEL SUDOKU
biggest, maxArea = biggestContour(contours) # BUSCAR EL CONTONOR MAS GRANDE
print("vertices")
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print("vertices ordenados")
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DINUJAR EL CONTORNO MAS GRANDE
    pts1 = np.float32(biggest) # PREPARA LOS PUNTOS DE LAS ESQUINAS
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARA LOS PUNTOS DE LAS ESQUINAS
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # TRANSFORMA LA PERSPECTIVA EN RELACION A CUATRO PUNTOS DE LA IMAGEN
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    #### 4. DIVIDIR LA IMAGEN Y ENCONTRAR CADA DIGITO DISPONIBLE
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print("Numero de cajas")
    print(len(boxes))
    # cv2.imshow("Sample",boxes[4])
    print("Probabilidades de ser digito")
    numbers = getPredection(boxes, model)
    print("Digitos encontrados en las cajas")
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print("Donde hallan numeros colocamos un 0 y donde no un 1")
    print(posArray)


    #### 5. ENCONTRAR SOLUCIÓN
    board = np.array_split(numbers,9)
    print("Array de 9x9")
    print("sudoku sin resolver")
    print(board)
    try:
        sudukoSolver.solve(board)
    except:
        pass
    print("Sudoku resuleto")
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # #### 6. PBREPONER LA SOLUCION EN UNA CIPIA DE LA IMAGEN ORIGINAL
    pts2 = np.float32(biggest) # PREPARAR LOS PUNTOS
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARAR LOS PUNTOS
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])


    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")

cv2.waitKey(0)