package com.example.thesis_application;
import java.util.List;

////Reprezentuje całą odpowiedź z API
//public class YoloResponse {
//    private List<Detection> detections;
//    private String image_base64;
//
//
//    public List<Detection> getDetections() {return detections;}
//    public void setDetections(List<Detection> detections) {this.detections = detections;}
//    public String getImageBase64() {
//            return image_base64;
//        };
//    public void setImageBase64(String image_base64) { this.image_base64 = image_base64; }
//}


import com.google.gson.annotations.SerializedName;

public class YoloResponse {
    private List<Detection> detections;

    @SerializedName("image_base64")
    private String imageBase64;

    public List<Detection> getDetections() { return detections; }
    public void setDetections(List<Detection> detections) { this.detections = detections; }

    public String getImageBase64() { return imageBase64; }
    public void setImageBase64(String imageBase64) { this.imageBase64 = imageBase64; }
}

