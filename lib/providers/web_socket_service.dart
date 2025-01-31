import 'dart:convert';
import 'package:web_socket_channel/io.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class WebSocketService {
  WebSocketChannel? channel;

  // Method to connect to the WebSocket
  void connect() {
    channel = IOWebSocketChannel.connect(
      Uri.parse('ws://127.0.0.1:8000/ws/eeg-stream'),
    );

    channel!.stream.listen((message) {
      print("Received EEG data: $message");
      _handleMessage(message);
    }, onError: (error) {
      print("Error: $error");
    }, onDone: () {
      print("Connection closed");
    });

    // Send message to start streaming from the first Muse device
    channel?.sink.add("start_streaming");
  }

  // Method to send a message to the WebSocket server
  void sendMessage(String message) {
    if (channel != null) {
      channel?.sink.add(message);
    }
  }

  // Method to request attention levels from the server
  void getAttentionLevel() {
    if (channel != null) {
      // Send the "get_attention_level" message to the server
      channel?.sink.add("get_attention_level");
    }
  }

  // Method to handle incoming messages
  void _handleMessage(String message) {
    try {
      var data = jsonDecode(message);

      // Check if the response contains attention levels
      if (data.containsKey('attention_levels')) {
        var attentionLevels = data['attention_levels'];

        // Process the attention levels
        print('Attention Levels: $attentionLevels');
      } else {
        print("Invalid response: $message");
      }
    } catch (e) {
      print("Error parsing message: $e");
    }
  }

  // Method to close the WebSocket connection
  void close() {
    channel?.sink.close();
  }
}
