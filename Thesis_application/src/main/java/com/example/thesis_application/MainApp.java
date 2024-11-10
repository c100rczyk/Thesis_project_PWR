package com.example.thesis_application;

import com.google.gson.JsonSyntaxException;
import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.scene.layout.BorderPane;
import javafx.scene.control.Button;
import javafx.scene.layout.HBox;
import javafx.scene.image.Image;

import javafx.scene.canvas.Canvas;          // dodanie planszy 2D
import javafx.scene.canvas.GraphicsContext; // dodanie planszy 2D
import javafx.scene.layout.StackPane;

import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.application.Platform;
import javafx.animation.AnimationTimer;
import javafx.scene.text.Font;

import javafx.scene.control.Label;          //panel boczny

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;                        //Integracja z API
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import com.google.gson.Gson;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.List;

import java.time.LocalDateTime;


public class MainApp extends Application {
    private Canvas canvas2D;
    private Canvas canvasSiamese;
    private Canvas cameraCanvas;
    private GraphicsContext gc;
    private HttpClient client;
    private Label siameseLabel1;
    private Label siameseLabel2;
    private Label siameseLabel3;
    private ImageView ObjectDetectionView;
    private ImageView cameraSiamese;

    @Override
    public void start(Stage primaryStage){
        BorderPane root = new BorderPane();
        //Pane root = new Pane();
        //AnchorPane root = new AnchorPane();

        Scene scene = new Scene(root, 1800, 900);
        primaryStage.setTitle("Thesis Application");
        primaryStage.setScene(scene);
        primaryStage.show();

        HBox topMenu = new HBox();
        Button btn_siamese = new Button("Siamese");
        btn_siamese.setFont(new Font("Arial", 20));
        topMenu.getChildren().add(btn_siamese);
        root.setTop(topMenu);

        // PLANSZA 2D_______________________________________________________________
        Label planszaLabel = new Label("Plansza wyświetlanjąca aktualną sytuację na scenie roboczej");
        planszaLabel.setFont(new Font("Arial", 25));

        canvas2D = new Canvas(830, 600);
        gc = canvas2D.getGraphicsContext2D();
        StackPane canvasContainer = new StackPane(canvas2D);
        canvasContainer.setStyle("-fx-border-color: #cc1717; -fx-border-width: 2;");
        canvasContainer.setMaxWidth(canvas2D.getWidth());
        canvasContainer.setMaxHeight(canvas2D.getHeight());

        VBox centerContainer = new VBox();
        centerContainer.getChildren().addAll(planszaLabel, canvasContainer);
        centerContainer.setAlignment(Pos.CENTER);
        centerContainer.setSpacing(10);

        //root.setCenter(centerContainer);    // ustawienie planszy w środkowej częsci aplikacji
        centerContainer.setLayoutX(450);
        centerContainer.setLayoutY(550);
        root.getChildren().add(centerContainer);


        //PRAWY DOLNY PANEL INTEL____________________________________
        ObjectDetectionView = new ImageView();
        ObjectDetectionView.setFitWidth(550);
        //cameraView.setFitHeight(650);
        ObjectDetectionView.setPreserveRatio(true);

        Label camera3D_label = new Label("Bierzący widok z kamery rgb, detekcja obiektów");
        camera3D_label.setFont(new Font("Arial", 25));
        centerContainer.setAlignment(Pos.CENTER);
        centerContainer.setSpacing(10);

        VBox rightPanel = new VBox();
        rightPanel.getChildren().addAll(ObjectDetectionView, camera3D_label);
        rightPanel.setLayoutX(1150);
        rightPanel.setLayoutY(400);
        //root.setRight(rightPanel);
        root.getChildren().add(rightPanel);

        //PRAWY GÓRNY PANEL SieciSjamskie___________________________
        cameraSiamese = new ImageView();
        cameraSiamese.setFitWidth(200);
        cameraSiamese.setPreserveRatio(true);

//        canvasSiamese = new Canvas(300, 400);
//        gc = canvasSiamese.getGraphicsContext2D();
//        StackPane container_with_siamese = new StackPane(canvasSiamese);
        
        Label cameraSiamese_label = new Label("Widok z kamery na chwytaku");
        cameraSiamese_label.setFont(new Font("Arial", 18));

        VBox rightPanelUp = new VBox();
        rightPanelUp.getChildren().addAll(cameraSiamese_label, cameraSiamese);
        rightPanelUp.setAlignment(Pos.CENTER);
        rightPanelUp.setSpacing(10);
        rightPanelUp.setLayoutX(1300);
        rightPanelUp.setLayoutY(150);

        root.getChildren().add(rightPanelUp);


        //PRAWY PANEL ETYKIETY SIECI SJAMSKIE :
        siameseLabel1 = new Label("Predykcja 1");
        siameseLabel2 = new Label("Predykcja 2");
        siameseLabel3 = new Label("Predykcja 3");

        siameseLabel1.setFont(new Font("Arial", 25));
        siameseLabel2.setFont(new Font("Arial", 25));
        siameseLabel3.setFont(new Font("Arial", 25));
        VBox rightPanelLabels = new VBox();
        rightPanelLabels.getChildren().addAll(siameseLabel1, siameseLabel2, siameseLabel3);
        //rightPanelLabels.setAlignment(Pos.TOP_RIGHT);
        rightPanelLabels.setSpacing(10);
        rightPanelLabels.setLayoutX(1300);
        rightPanelLabels.setLayoutY(100);
        root.setRight(rightPanelLabels);


        //____________________________________________________

        //Tworzenie klienta HTTP
        client = HttpClient.newHttpClient();     //umożliwia wysyłanie żądań HTTP

        //____________________________________________________

        //TU był kod *12*

        //cameraCanvas = new Canvas(200, 200);
        StackPane imageStack = new StackPane();
        imageStack.getChildren().addAll(ObjectDetectionView);//, cameraCanvas);
        rightPanel.getChildren().add(imageStack);


        //AKCJE:_____________

        // Po naciśnięciu zwracana jest lista 3 etykiet które wykrył model sieci Sjamskich
        btn_siamese.setOnAction(event ->{
            try {
                callSiameseModel();
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        // Aktualizacja DANYCH i ZDJĘĆ w aplikacji
        AnimationTimer timer = new AnimationTimer() {
        @Override
        public void handle(long now) {
            try {
                updateImageObjectDetectionView();  // Aktualizujemy zdjęcie rgb Detekcji Obiektów i planszę
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            LocalDateTime currentDateTime = LocalDateTime.now();
            System.out.println("Aktualny czas: "+ currentDateTime);
        }
        };
        timer.start();
    }



    // PLANSZA 2D ______________________________________________________________________________________
    // AKTUALIZACJA DANYCH NA PLANSZY 2D


    // POLE WIZUALIZACJ PLANSZY 2D
    private void updateCanvas(GraphicsContext gc, List<Detection> detections){
        gc.clearRect(0, 0, canvas2D.getWidth(), canvas2D.getHeight());
        for(Detection det : detections){
            double canvasX = (det.getX_min() + det.getX_max())/2;     //szerokość w aplikacji
            double szerokosc_na_planszy = RangeMapper.mapValue(canvasX, 0, 640, 0, 830);
            double glebokosc = RangeMapper.mapValue(det.getDepth(), 1, 2.3, 0, 600);
            double glebokosc_na_planszy = - glebokosc + 600; //głębokość w aplikacji
            gc.fillOval(szerokosc_na_planszy, glebokosc_na_planszy, 20, 20);  // kółko reprezentujące wykryty obiekt
            gc.fillText(det.getLabel(), szerokosc_na_planszy+10, glebokosc_na_planszy);
        }
    }



    // OBJECT DETECTION VIEW_______________________________________________________________________

    private void updateImageObjectDetectionView() throws IOException, InterruptedException {
        String objectDetectionEndPoint = "http://localhost:8000/object_detection";

        HttpRequest requestObjectDetection = HttpRequest.newBuilder(URI.create(objectDetectionEndPoint))
                .POST(HttpRequest.BodyPublishers.noBody()).
                build();

        HttpResponse<String> response = client.send(requestObjectDetection, HttpResponse.BodyHandlers.ofString());

        String responseBody = response.body();

//        System.out.println("Status Code: " + response.statusCode());
//        System.out.println("Response Body: " + responseBody);

        try {
            Gson gson = new Gson();
            YoloResponse yolo_data = gson.fromJson(responseBody, YoloResponse.class);

            //WSPÓŁRZĘDNE NA PLANSZY
            Platform.runLater(() -> {
                updateCanvas(gc, yolo_data.getDetections());
            });

            //ZDJĘCIE
            Platform.runLater(() -> {
                updateImageFromBase64(ObjectDetectionView, yolo_data.getImageBase64());
            });
        } catch (JsonSyntaxException e) {
            System.out.println("Błąd podczas parsowania JSON: " + e.getMessage());
            System.out.println("Response Body: " + responseBody);
        }

    }




    // SIECI SJAMSKIE ________________________________________________________________________________
    // POBIERANIE DANYCH Z MODELU SIECI SJAMSKICH
    private void callSiameseModel() throws IOException, InterruptedException {
        String siameseEndPoint = "http://localhost:8000/predict_siamese";

        HttpRequest requestSiamese = HttpRequest.newBuilder(URI.create(siameseEndPoint))
                .POST(HttpRequest.BodyPublishers.noBody())
                .build();

        HttpResponse<String> response = client.send(requestSiamese, HttpResponse.BodyHandlers.ofString());

        //Przetworzenie odpowiedzi JSON
        String responseBody = response.body();

        Gson gson = new Gson();
        SiameseResponse siameseData = gson.fromJson(responseBody, SiameseResponse.class);

        //ETYKIETY
        Platform.runLater(() -> {
            updateSiameseLabels(siameseData);
        });
        //OBRAZ
        Platform.runLater(() -> {
            updateImageFromBase64(cameraSiamese, siameseData.getImageBase64());
        });
    }

    private void updateImageFromBase64(ImageView cameraView, String base64Image) {
        byte[] imageBytes = Base64.getDecoder().decode(base64Image);
        InputStream is = new ByteArrayInputStream(imageBytes);
        Image image = new Image(is);
        cameraView.setImage(image);
    };
    private void updateSiameseLabels(SiameseResponse siameseData){
    List<String> labels = siameseData.getLabels();
    siameseLabel1.setText(labels.size() > 0 ? labels.get(0) : "brak danych");
    siameseLabel2.setText(labels.size() > 1 ? labels.get(1) : "brak danych");
    siameseLabel3.setText(labels.size() > 2 ? labels.get(2) : "brak danych");
    }




    public static void main(String[] args) {
        launch(args);
    }


}




//        String testResponse = "{ \"labels\": [\"Pomarancza\", \"Mandarynka\", \"Banan\"] }";
//        Gson gson = new Gson();
//        SiameseResponse siamese_data = gson.fromJson(testResponse, SiameseResponse.class);
//        Platform.runLater(() -> {
//            updateSiameseLabels(siamese_data);
//        });
//        updateImageSiameseView();
//    }


//    private void updateImageSiameseView(){
//        Platform.runLater(() -> {
//            File image_from_siamese = new File("/home/c100rczyk/Projekty/Thesis_project_PWR/data/images/Pomarancza.jpg");
//            if (image_from_siamese.exists()) {
//            String imagePath = image_from_siamese.toURI().toString();
//            Image image = new Image(imagePath);
//            cameraSiamese.setImage(image);
//        } else {
//            System.out.println("Plik nie istnieje: " + image_from_siamese.getAbsolutePath());
//        }
//        });
//    }

//    private void updateImageObjectDetectionView(){
//    Platform.runLater(() -> {
//        //String imageURL = "https://jsonplaceholder.typicode.com/photos/17";
//        File imageFile = new File("/home/c100rczyk/Projekty/Thesis_project_PWR/data/images/image_good.png");
//        if (imageFile.exists()) {
//            String imagePath = imageFile.toURI().toString();
//            Image image = new Image(imagePath);
//            ObjectDetectionView.setImage(image);
//        } else {
//            System.out.println("Plik nie istnieje: " + imageFile.getAbsolutePath());
//            }
//        });
//    }

//    private void update_plansza2D_data() {
//        // Przykładowe dane testowe
//        String testResponse = "{\"detections\": ["
//            + "{\"label\": \"person\", \"x\": 100, \"y\": 800, \"depth\": 590},"
//            + "{\"label\": \"car\", \"x\": 300, \"y\": 10, \"depth\": 10}"
//            + "]}";
//
//        Gson gson = new Gson();
//        YoloResponse yolo_data = gson.fromJson(testResponse, YoloResponse.class);
//
//        Platform.runLater(() -> {
//            updateCanvas(gc, yolo_data.getDetections());
//        });
//    }

//*12*______
        //String yoloEndPoint = "http://localhost:8000/yolo";
//        String yoloEndPoint = "https://jsonplaceholder.typicode.com/posts/1";
//        HttpRequest request_to_yolo = HttpRequest.newBuilder(URI.create(yoloEndPoint)).GET().build();
//        client.sendAsync(request_to_yolo, HttpResponse.BodyHandlers.ofString())
//                .thenApply(HttpResponse::body)
//                .thenAccept(response -> {
//            //przetwarzanie odpowiedzi
//            String testResponse = "{\"detections\": ["
//                + "{\"label\": \"person\", \"x\": 100, \"y\": 800, \"depth\": 590},"
//                + "{\"label\": \"car\", \"x\": 300, \"y\": 10, \"depth\": 10}"
//                + "]}";
//            Gson gson = new Gson();
//            YoloResponse yolo_data = gson.fromJson(testResponse, YoloResponse.class);
//
//            Platform.runLater(() -> {       // aktualizacja interfejsu użytkownika w wątku javaFX
//                updateCanvas(gc, yolo_data.getDetections());
//            });
//
//        }).exceptionally(e->{
//            e.printStackTrace();
//            return null;
//        });



        //Pobieranie obrazu z FAST API (zdjęcie z kamery Intel - resize do rozmiaru który obsługuje też YOLO).
        // ZDJĘCIE YOLO

//        String imageEndPoint = "https://jsonplaceholder.typicode.com/photos/17";
//        HttpRequest imageRequest = HttpRequest.newBuilder(URI.create(imageEndPoint)).GET().build();
//
//        client.sendAsync(imageRequest, HttpResponse.BodyHandlers.ofString())
//                .thenApply(HttpResponse::body)
//                .thenAccept(response -> {
//                    Platform.runLater(() -> {
//                        //updateImageView(cameraView, response);
//                        try {
//                            updateImageObjectDetectionView();
//                        } catch (IOException e) {
//                            throw new RuntimeException(e);
//                        } catch (InterruptedException e) {
//                            throw new RuntimeException(e);
//                        }
//                    });
//                }).exceptionally(e-> {
//                    e.printStackTrace();
//                    return null;
//                });
//*12*