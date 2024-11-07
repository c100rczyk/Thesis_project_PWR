import com.google.gson.JsonArray;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.IOException;


public class Get_Object_Detection {
    public static void main(String[] args){
        CloseableHttpClient httpClient = HttpClients.createDefault();

        String url = "http://localhost:5000/predict_siamase";

        HttpPost httpPost = new HttpPost(url);

        try{
            CloseableHttpResponse response = httpClient.execute(httpPost);
            int statusCode = response.getCode();

            if(statusCode == 200){
                String responseBody = EntityUtils.toString(response.getEntity());
                System.out.println("Odpowiedź z serwera Sieci Syjamskich: " + responseBody);

                // parsowanie odpowiedzi JSON
                Gson gson = new Gson();
                JsonObject jsonResponse = gson.fromJson(responseBody, JsonObject.class);

                JsonArray labelsArray = jsonResponse.getAsJsonArray("labels");
                for (int i = 0; i < labelsArray.size(); i++) {
                    String label = labelsArray.get(i).getAsString();
                    System.out.println("Etykieta " + (i + 1) + ": " + label);
                }

            }
            else{
                System.err.println("Błąd, otrzymano kod status: " + statusCode);
            }



        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }
}
