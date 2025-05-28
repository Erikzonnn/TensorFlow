let modelo;
const IMG_WIDTH = 180;
const IMG_HEIGHT = 180; 

const NOMBRES_CLASES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'];

const modelPath = './carpeta_salida_tfjs_flores/model.json'; // Ruta relativa al model.json

async function cargarModelo() {
    console.log("app.js: Intentando cargar modelo desde:", modelPath);
    const resultadoElement = document.getElementById('resultado');
    resultadoElement.innerText = 'Cargando modelo...';

    try {
        modelo = await tf.loadLayersModel(modelPath);
        console.log("Modelo cargado exitosamente:", modelo);
        resultadoElement.innerText = 'Modelo cargado. Sube una imagen para clasificar.';

        // Calentamiento del modelo (opcional, para la primera predicción más rápida)
        console.log("Calentando el modelo...");
        const dummyInput = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3]);
        const dummyPrediction = modelo.predict(dummyInput);
        dummyPrediction.dispose(); // Descartar la predicción dummy
        dummyInput.dispose();    // Descartar la entrada dummy
        console.log("Modelo calentado.");

    } catch (error) {
        console.error("Error al cargar el modelo:", error);
        // Mostrar el error completo en la consola es útil.
        // A veces el error puede tener más detalles si lo inspeccionas.
        console.dir(error);
        resultadoElement.innerText = `Error al cargar el modelo. Revisa la consola del navegador (F12). Detalles: ${error.message}`;
    }
}

function mostrarPreview(event) {
    const reader = new FileReader();
    const output = document.getElementById('preview');
    const resultadoElement = document.getElementById('resultado');

    reader.onload = function(){
        output.src = reader.result;
        output.style.display = 'block'; // Asegurarse de que la imagen sea visible
    };

    if (event.target.files && event.target.files[0]) {
        reader.readAsDataURL(event.target.files[0]);
        if (modelo) {
            resultadoElement.innerText = 'Imagen cargada. Presiona "Clasificar Flor".';
        } else {
            resultadoElement.innerText = 'Imagen cargada. Esperando carga del modelo...';
        }
    } else {
        output.src = "#"; // Limpiar si no hay archivo
        output.style.display = 'none'; // Ocultar si no hay imagen
        if (modelo) {
            resultadoElement.innerText = 'Modelo cargado. Sube una imagen para clasificar.';
        } else {
            resultadoElement.innerText = 'Cargando modelo...';
        }
    }
}

async function predecirFlor() {
    const imgElement = document.getElementById('preview');
    const inputFile = document.getElementById('selectorImagen');
    const resultadoElement = document.getElementById('resultado');

    if (!inputFile.files || inputFile.files.length === 0) {
        alert("Por favor, selecciona una imagen primero.");
        return;
    }
    if (!modelo) {
        alert("El modelo aún no se ha cargado completamente. Por favor, espera o revisa la consola por errores.");
        return;
    }

    resultadoElement.innerText = 'Clasificando...';
    console.log("Iniciando predicción...");

    let tensorImg; // Declarar fuera del try para poder hacer dispose en el catch

    try {
        // tf.tidy ayuda a manejar la memoria de los tensores intermedios
        tensorImg = tf.tidy(() => {
            console.log("Preprocesando imagen...");
            let img = tf.browser.fromPixels(imgElement); // Crea un tensor desde el elemento <img>
            // Redimensionar la imagen a las dimensiones esperadas por el modelo
            img = tf.image.resizeNearestNeighbor(img, [IMG_HEIGHT, IMG_WIDTH]);
            img = img.toFloat(); // Convertir a float
            // Normalizar los píxeles al rango [0, 1] si así entrenaste tu modelo en Python
            img = img.div(tf.scalar(255.0));
            // Añadir una dimensión de batch (el modelo espera [batch_size, alto, ancho, canales])
            return img.expandDims();
        });

        console.log("Tensor de entrada creado, forma:", tensorImg.shape);

        const prediccionesTensor = modelo.predict(tensorImg); // Realizar la predicción
        const prediccionesArray = await prediccionesTensor.data(); // Obtener los datos como un array

        console.log("Predicciones (raw):", prediccionesArray);

        // Encontrar el índice de la clase con la mayor probabilidad
        let indiceMaxProb = 0;
        let maxProb = prediccionesArray[0];
        for (let i = 1; i < prediccionesArray.length; i++) {
            if (prediccionesArray[i] > maxProb) {
                maxProb = prediccionesArray[i];
                indiceMaxProb = i;
            }
        }

        if (indiceMaxProb < NOMBRES_CLASES.length) {
            const clasePredicha = NOMBRES_CLASES[indiceMaxProb];
            const confianza = (maxProb * 100).toFixed(2);
            resultadoElement.innerText = `Predicción: ${clasePredicha} (${confianza}%)`;
        } else {
            console.error("Índice de predicción fuera de rango para NOMBRES_CLASES. ¿Coincide el número de clases?");
            resultadoElement.innerText = `Error: Índice de predicción (${indiceMaxProb}) fuera de rango.`;
        }

        // Liberar memoria de los tensores
        tensorImg.dispose();
        prediccionesTensor.dispose();
        console.log("Tensores liberados.");

    } catch (error) {
        console.error("Error durante la predicción:", error);
        console.dir(error);
        resultadoElement.innerText = `Error al realizar la predicción. Revisa la consola. Detalles: ${error.message}`;
        if (tensorImg) { // Asegurarse de liberar memoria si hubo error y el tensor se creó
            tensorImg.dispose();
            console.log("Tensor de entrada liberado después de error.");
        }
    }
}

// Iniciar la carga del modelo cuando se carga la ventana
window.onload = cargarModelo;