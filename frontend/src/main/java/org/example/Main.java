package org.example;

import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

public class Main {
    public static void main(String[] args) {
        CloseableHttpClient httpClient = HttpClients.createDefault();
        String url = "https://httpbin.org/post";

        // Tworzenie żądania POST
        HttpPost postRequest = new HttpPost(url);

        // Tworzenie obiektu JSON do wysłania
        JsonObject json = new JsonObject();
        json.addProperty("name", "Jan Kowalski");
        json.addProperty("email", "jan.kowalski@example.com");

        // Konwersja obiektu JSON do String
        String jsonString = new Gson().toJson(json);

        try {
            // Ustawienie treści żądania
            StringEntity requestEntity = new StringEntity(
                    jsonString,
                    org.apache.hc.core5.http.ContentType.APPLICATION_JSON);
            postRequest.setEntity(requestEntity);

            // Wykonanie żądania
            CloseableHttpResponse response = ((CloseableHttpClient) httpClient).execute(postRequest);

            // Odczytanie statusu i treści odpowiedzi
            int statusCode = response.getCode();
            System.out.println("Status code: " + statusCode);

            String responseBody = EntityUtils.toString(response.getEntity());
            System.out.println("Response body: " + responseBody);

            // Zamknięcie odpowiedzi
            response.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                // Zamknięcie klienta HTTP
                httpClient.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}