package com.example.thesis_application;

public class Detection {
// Reprezentuje pojedyncze wykrycie obiektu
    private double x_min;
    private double y_min;
    private double x_max;
    private double y_max;
    private String label;
    private double confidence;
    private double depth;
    //gettery i settery


    public double getX_min() { return x_min; }
    public void setX_min(double x_min) { this.x_min = x_min; }

    public double getY_min() { return y_min; }
    public void setY_min(double y_min) { this.y_min = y_min; }

    public double getX_max() { return x_max; }
    public void setX_max(double x_max) { this.x_max = x_max; }

    public double getY_max() { return y_max; }
    public void setY_max(double y_max) { this.y_max = y_max; }

    public String getLabel() { return label; }
    public void setLabel(String label) { this.label = label; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }

    public double getDepth() {return depth;}
    public void setDepth(double depth) { this.depth = depth; }
}

//package com.example.thesis_application;
//
//public class Detection {
//// Reprezentuje pojedyncze wykrycie obiektu
//    private double x;
//    private double y;
//    private double depth;
//    private String label;
//    private double confidence;
//    //gettery i settery
//
//
//    public String getLabel() {return label;}
//    public void setLabel(String label) {this.label = label;}
//
//    public double getX() {return x;}
//    public void setX(double x){this.x = x;}
//
//    public double getY() {return y;}
//    public void setY(double y) {this.y = y;}
//
//    public double getDepth() {return depth;}
//    public void setDepth(double depth) {this.depth = depth;}
//
//    public double getConfidence() { return confidence; }
//    public void setConfidence(double confidence) { this.confidence = confidence; }
//}
