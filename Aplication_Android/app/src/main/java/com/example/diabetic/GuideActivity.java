package com.example.diabetic;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.text.util.Linkify;
import android.widget.TextView;

public class GuideActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_guide);

        TextView youtubeLinkEng = findViewById(R.id.youtubeLinkEnglish);
        TextView youtubeLinkInd = findViewById(R.id.youtubeLinkIndonesia);
        TextView pageGuideToMain = findViewById(R.id.clickToMain);

        String youtubeUrl = "https://youtu.be/basRkp_VOvQ"; // Ganti dengan URL video YouTube Anda
        youtubeLinkEng.setText(youtubeUrl);
        youtubeLinkInd.setText(youtubeUrl);

        // Otomatis membuat URL menjadi link
        Linkify.addLinks(youtubeLinkEng, Linkify.WEB_URLS);
        Linkify.addLinks(youtubeLinkInd, Linkify.WEB_URLS);

        // Set button to active gallery view click action, and it used for assign request code
        pageGuideToMain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(GuideActivity.this, MainActivity.class);
                startActivity(intent);
                finish();
            }
        });
    }
}