package com.example.thesis_application;

public class Detection {
// Reprezentuje pojedyncze wykrycie obiektu
    private double x;
    private double y;
    private double depth;
    private String label;
    private double confidence;
    //gettery i settery


    public String getLabel() {return label;}
    public void setLabel(String label) {this.label = label;}

    public double getX() {return x;}
    public void setX(double x){this.x = x;}

    public double getY() {return y;}
    public void setY(double y) {this.y = y;}

    public double getDepth() {return depth;}
    public void setDepth(double depth) {this.depth = depth;}

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
}
