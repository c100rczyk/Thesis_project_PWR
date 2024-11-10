package com.example.thesis_application;
import java.awt.geom.AffineTransform;

public class RangeMapper {

    // Metoda do przekształcenia wartości z zakresu (a, b) na zakres (c, d)
    public static double mapValue(double value, double a, double b, double c, double d) {
        // Tworzymy transformację liniową
        AffineTransform transform = new AffineTransform();
        transform.setToScale((d - c) / (b - a), 1);
        double offset = c - a * (d - c) / (b - a);
        return value * transform.getScaleX() + offset;
    }
}