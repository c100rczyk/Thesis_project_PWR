module com.example.thesis_application {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.kordamp.ikonli.javafx;
    requires java.net.http;
    requires com.google.gson;

    opens com.example.thesis_application to javafx.fxml, com.google.gson;
    exports com.example.thesis_application;
}