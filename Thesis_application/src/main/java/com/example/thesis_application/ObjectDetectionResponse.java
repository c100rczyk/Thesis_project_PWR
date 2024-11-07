package com.example.thesis_application;
import java.util.List;

public class ObjectDetectionResponse {
    private List<String> labels;
    private String image_base64;

    public List<String> getLabels() { return labels; }
    public void setLabels(List<String> labels) { this.labels = labels; }
    public String getImageBase64() {
            return image_base64;
        }
}
