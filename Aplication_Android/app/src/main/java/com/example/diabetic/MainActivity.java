package com.example.diabetic;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.service.controls.templates.ThumbnailTemplate;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.diabetic.ml.ModelFix;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    private Button selectBtn, captureBtn, predictBtn;
    private TextView result;
    private ImageView imageView;
    Bitmap bitmap;

    @Override
    // This function called once when the program created
    protected void onCreate(Bundle savedInstanceState) {
        /*
        REQUEST CODE
        10 - Open Gallery View - Activity
        11 - Camera permission - RequestPermission
        12 - Open Camera View - Activity
         */
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        selectBtn = findViewById(R.id.selectBtn);
        captureBtn = findViewById(R.id.captureBtn);
        predictBtn = findViewById(R.id.predictBtn);
        result = findViewById(R.id.result);

        // Set button to active gallery view click action, and it used for assign request code
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });

        // Set button to active camera view click action, and it used for assign request code
        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });

        // Set button prediction click action, doesn't set request code
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String text = new String();
                String[] classes = {"DR", "No DR"};
                int pred = 0;

                // This is for try catch if there was an image or it wasn't
                try {
                    pred = bitmap.getByteCount();
                }
                catch (Exception e) {
                    result.setText("Choose the image");
                }

                if (pred > 0) {
                    // Detection process
                    try {
                        ModelFix model = ModelFix.newInstance(MainActivity.this);

                        // Creates inputs for reference
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                        // Prepare for buffer image
                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
                        byteBuffer.order(ByteOrder.nativeOrder());

                        // Match the data pixel into the buffer
                        int [] intValues = new int [224 * 224];
                        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getHeight(), bitmap.getHeight());
                        int pixel = 0;
                        for (int i = 0; i < 224; i++) {
                            for (int j = 0; j < 224; j++) {
                                int val = intValues[pixel++];
                                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                                byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                            }
                        }

                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        ModelFix.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        // Get prediction and confidence
                        float [] confidences = outputFeature0.getFloatArray();
                        int maxPos = 0;
                        float maxConfidence = 0;
                        for (int i = 0; i < confidences.length; i++) {
                            if (confidences[i] > maxConfidence) {
                                maxConfidence = confidences[i];
                                maxPos = i;
                            }
                        }

                        // Set the output of prediction
                        text = classes[maxPos].concat("\n").concat(String.valueOf(maxConfidence * 100)).concat("%");
                        result.setText(text);

                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                        // TODO Handle the exception
                    }
                }
            }
        });

        // Call function for get permission
        getPermission();
    }

    void getPermission() {
        // Checking android version
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            // Checking camera permission if permission not given yet
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                captureBtn.setEnabled(false);
                ActivityCompat.requestPermissions(MainActivity.this, new String[] {Manifest.permission.CAMERA}, 11);
            }
        }
    }

    @Override
    // Request permission function
    public void onRequestPermissionsResult(int requestCode, @Nullable String [] permission, @Nullable int [] grantResult) {
        super.onRequestPermissionsResult(requestCode, permission, grantResult);
        // Request for camera permission (request code 11)
        if (requestCode == 11) {
            if (grantResult.length > 0 && grantResult[0] == PackageManager.PERMISSION_GRANTED) {
                this.getPermission();
                captureBtn.setEnabled(true);
            }
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            // Trigger when button Choose Image clicked and requestcode change 10
            if (requestCode == 10 && data != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            // Trigger when button Capture Image clicked and requestcode change 12
            if (requestCode == 12 && data != null) {
                bitmap = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(bitmap.getWidth(), bitmap.getHeight());
                bitmap = ThumbnailUtils.extractThumbnail(bitmap, dimension, dimension);
                bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false);
                imageView.setImageBitmap(bitmap);
            }
        }
    }
}