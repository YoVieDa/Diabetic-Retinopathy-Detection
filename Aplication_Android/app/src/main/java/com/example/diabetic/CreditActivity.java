package com.example.diabetic;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import androidx.appcompat.app.AppCompatActivity;

public class CreditActivity extends AppCompatActivity {
    private static int TIME_OUT = 2500; // Waktu penundaan dalam milidetik (misalnya 4 detik)

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_credit); // Pastikan layout yang benar

        // Menggunakan Handler untuk menunda pindah activity
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                // Membuka MainActivity
                Intent intent = new Intent(CreditActivity.this, GuideActivity.class);
                startActivity(intent);
                finish(); // Menutup CreditActivity agar tidak bisa kembali ke activity ini
            }
        }, TIME_OUT);
    }
}