#include <Arduino.h>
#include <OV767X_TinyMLx.h>
// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Include the model
#include "custom_cnn_gray_model_quantized.h" // Your quantized model in the include folder

// Camera settings
#define IMAGE_WIDTH 176    // QCIF resolution from camera
#define IMAGE_HEIGHT 144   // QCIF resolution from camera
#define CHANNELS 1         // Grayscale 
#define MODEL_WIDTH 96     // Model input width
#define MODEL_HEIGHT 96    // Model input height
#define MODEL_CHANNELS 1   // Grayscale

// Inference settings
#define THRESHOLD 0.5      // Detection threshold

// Buffer for the camera image
uint8_t image_buffer[IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];
uint8_t resized_buffer[MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS];

// TensorFlow Lite model
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays
constexpr int kTensorArenaSize = 100 * 1024;
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

// Simple resize function (nearest neighbor)
void resizeImage(uint8_t* input, uint8_t* output, int inputWidth, int inputHeight, 
                int outputWidth, int outputHeight, int channels) {
  float x_ratio = ((float)inputWidth - 1) / outputWidth;
  float y_ratio = ((float)inputHeight - 1) / outputHeight;
  
  for (int y = 0; y < outputHeight; y++) {
    int y2 = (int)(y * y_ratio);
    for (int x = 0; x < outputWidth; x++) {
      int x2 = (int)(x * x_ratio);
      if (channels == 1) {
        output[(y * outputWidth + x)] = input[(y2 * inputWidth + x2)];
      }
      // If you need to handle RGB to grayscale conversion
      else if (channels == 3) {
        int rgb_index = (y2 * inputWidth + x2) * 3;
        int r = input[rgb_index];
        int g = input[rgb_index + 1];
        int b = input[rgb_index + 2];
        // Standard grayscale conversion formula
        output[(y * outputWidth + x)] = (uint8_t)(0.299*r + 0.587*g + 0.114*b);
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000);  // Wait for serial but continue after 5 seconds
  
  Serial.println("Car Detection with Arduino Nano 33 BLE Sense");
  
  // Initialize the camera with RGB color
  if (!Camera.begin(QCIF, GRAYSCALE, 1, 0)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  Serial.println("Camera initialized");
  
  // Initialize TFLite model
  model = tflite::GetModel(custom_cnn_gray_model_quantized); // variable name from your .h file
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  Serial.println("Model loaded successfully");

  // Set up interpreter with operations commonly used in image classification models
  static tflite::MicroMutableOpResolver<11> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  // Build the interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory for the tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    Serial.print("Input tensor type: ");
    if (input->type == kTfLiteUInt8) Serial.println("UInt8");
    else if (input->type == kTfLiteFloat32) Serial.println("Float32");
    else if (input->type == kTfLiteInt8) Serial.println("Int8");
    else Serial.println("Other");
    while (1);
  }

  // Get pointers to the model's input and output tensors AFTER allocation
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.print("Input shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Output tensor dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
  
  Serial.println("Model initialized");
  
  // Debug model details more thoroughly
  Serial.print("Input details - type: ");
  if (input->type == kTfLiteUInt8) Serial.println("UInt8");
  else if (input->type == kTfLiteFloat32) Serial.println("Float32");
  else if (input->type == kTfLiteInt8) Serial.println("Int8");
  else Serial.println("Other");
    
  Serial.print("Input quantization - scale: ");
  Serial.print(input->params.scale);
  Serial.print(", zero_point: ");
  Serial.println(input->params.zero_point);
    
  Serial.print("Output details - type: ");
  if (output->type == kTfLiteUInt8) Serial.println("UInt8");
  else if (output->type == kTfLiteFloat32) Serial.println("Float32");
  else if (output->type == kTfLiteInt8) Serial.println("Int8");
  else Serial.println("Other");
    
  Serial.print("Output quantization - scale: ");
  Serial.print(output->params.scale);
  Serial.print(", zero_point: ");
  Serial.println(output->params.zero_point);
}

void loop() {
  Serial.println("Capturing image...");
  
  // Get image from camera
  Camera.readFrame(image_buffer);
  
  // You can comment out or conditionally print this large data block
  // to avoid flooding the serial monitor
  /*
  Serial.println("FULL IMAGE DATA START");
  for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
    Serial.print(image_buffer[i]);
    Serial.print(" ");
    if (i % 20 == 19) Serial.println();  // Add newline every 20 values for readability
  }
  Serial.println("\nFULL IMAGE DATA END");
  */

  // Resize the image to match model input size (96x96)
  resizeImage(image_buffer, resized_buffer, IMAGE_WIDTH, IMAGE_HEIGHT, 
              MODEL_WIDTH, MODEL_HEIGHT, CHANNELS);
  
  // Prepare input tensor based on the model type
  if (input->type == kTfLiteFloat32) {
    // For float model, normalize to [0,1]
    for (int i = 0; i < MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS; i++) {
      input->data.f[i] = resized_buffer[i] / 255.0f;
    }
  } else if (input->type == kTfLiteUInt8) {
    // For uint8 quantized model
    for (int i = 0; i < MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS; i++) {
      input->data.uint8[i] = resized_buffer[i];
    }
  } else if (input->type == kTfLiteInt8) {
    // For int8 quantized model
    for (int i = 0; i < MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS; i++) {
      // Convert uint8 to int8 using the input quantization parameters
      input->data.int8[i] = static_cast<int8_t>(resized_buffer[i] - 128);
    }
  }
  
  // Run inference
  Serial.println("Running inference...");
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }
  
  // Process the inference results
  float car_score = 0.0f;
  float no_car_score = .0f;

  // Extract result based on model output type
  if (output->type == kTfLiteUInt8) {
    // Debug raw output values
    Serial.print("Raw output values: ");
    for (int i = 0; i < 2; i++) {
      Serial.print(output->data.uint8[i]);
      Serial.print(" ");
    }
    Serial.println();
    
    uint8_t no_car_raw = output->data.uint8[0];
    uint8_t car_raw = output->data.uint8[1];
    
    // Convert from quantized value to float using output parameters
    no_car_score = (no_car_raw - output->params.zero_point) * output->params.scale;
    car_score = (car_raw - output->params.zero_point) * output->params.scale;
  } else if (output->type == kTfLiteInt8) {
    // Debug raw output values
    Serial.print("Raw output values: ");
    for (int i = 0; i < 2; i++) {
      Serial.print((int)output->data.int8[i]);
      Serial.print(" ");
    }
    Serial.println();
    
    int8_t no_car_raw = output->data.int8[0];
    int8_t car_raw = output->data.int8[1];
    
    // Convert from quantized value to float using output parameters
    no_car_score = (no_car_raw - output->params.zero_point) * output->params.scale;
    car_score = (car_raw - output->params.zero_point) * output->params.scale;
  } else if (output->type == kTfLiteFloat32) {
    no_car_score = output->data.f[0];
    car_score = output->data.f[1];
  }
  
  Serial.print("No car score: ");
  Serial.println(no_car_score, 6);
  Serial.print("Car score: ");
  Serial.println(car_score, 6);
  
  if (car_score > no_car_score && car_score > THRESHOLD) {
    Serial.println("CAR DETECTED!");
  } else {
    Serial.println("No car detected");
  }
  
  // Wait before next detection
  delay(2000);
}