//
//  Service.swift
//  csc2228
//
//  Created by Yolanda He on 2017-10-10.
//  Copyright © 2017 csc2228. All rights reserved.
//

import Foundation
import CoreMotion
import CoreLocation
import AWSDynamoDB

class Service {
    
    var samplingAccTimer: Timer!
    var samplingLocTimer: Timer!
    var uploadTimer: Timer!
    var action:String!
    var deviceName: String!
    var currentTime: String!
    var accQueue=[[AWSDynamoDBWriteRequest]]()
    var locQueue=[[AWSDynamoDBWriteRequest]]()
    var recordAcc=[AWSDynamoDBWriteRequest]()
    var recordLocation=[AWSDynamoDBWriteRequest]()
    var db = AWSDynamoDB.default()
    var updateInput:AWSDynamoDBUpdateItemInput!
    
    var motionManager: CMMotionManager!
    var locationManager:  CLLocationManager!
    
    var msgRetrievedCnt=0
    var msgPushInCnt = 0
    
    func newTimerTask(action: String!) {
        getActiontime()
        print("Creating new timer "+action)
        print("Action Time:"+currentTime)
        self.action=action
       
        getDeviceId()
        
        // Clean the existing timer first. That means, ensure the running timer will be clean out before a new task.
        clearTimers()
        
        //initialize the managers
        initManagers()
        
        //initialize AWS DynamoDB
        initDynamoDB()
        
        // Set up the timer to upload local dataset, and to sample data.
        uploadTimer = Timer.scheduledTimer(timeInterval: Constants.INTERVAL_UPLOAD, target: self, selector: #selector(sendExistingData), userInfo: nil, repeats: true)
        samplingAccTimer = Timer.scheduledTimer(timeInterval: Constants.INTERVAL_SAMPLING_ACCELEROMETER, target: self, selector: #selector(sampleAccelerometer), userInfo: nil, repeats: true)
        samplingLocTimer = Timer.scheduledTimer(timeInterval: Constants.INTERVAL_SAMPLING_LOCATION, target: self, selector: #selector(sampleLocation), userInfo: nil, repeats: true)
        
        

    }
    
    func initDynamoDB(){
        db = AWSDynamoDB.default()
    }
    
    
    
    func initManagers(){
        
        self.motionManager = CMMotionManager()
        self.motionManager.accelerometerUpdateInterval=Constants.INTERVAL_SAMPLING_ACCELEROMETER
        self.motionManager.startAccelerometerUpdates()
        
        
        if(CLLocationManager.locationServicesEnabled()){
            locationManager=CLLocationManager()
            locationManager.requestAlwaysAuthorization()
            locationManager.requestWhenInUseAuthorization()
            locationManager.desiredAccuracy = kCLLocationAccuracyBest
            locationManager.startUpdatingLocation()
        }
        else{
            print("location service unavailable")
        }
        
    }
    
    func cleanAll(){
        clearManagers()
        clearTimers()
        cleanCache()
    }
    
    func clearManagers(){
        guard let _=motionManager else {
            return
        }
        
        motionManager.stopAccelerometerUpdates()
        guard let _=locationManager else {
            return
        }
        
        locationManager.stopUpdatingLocation()
        
    }
    
    func clearTimers() {
        clearAccTimer()
        clearUploadTimer()
        clearLocTimer()
    }
    
    func clearLocTimer() {
        guard let _=samplingLocTimer else {
            return
        }
        
        print("samplingLocTimer Canceled")
        samplingLocTimer.invalidate()
    }
    func clearAccTimer()  {
        guard let _=samplingAccTimer else {
            return
        }
        
        print("samplingAccTimer Canceled")
        samplingAccTimer.invalidate()
    }
    
    func clearUploadTimer(){
        guard let _=uploadTimer else {
            return
        }
        
        print("uploadTimer Canceled")
        uploadTimer.invalidate()
    }
    
    
    func doesAccExceedMaxQueueSize(){
        if recordAcc.count == Constants.BATCH_SIZE {
            accQueue.append(recordAcc)
            recordAcc.removeAll()
        }
        

    
    }
    
    func doesLocExceedMaxQueueSize(){
//        print("array size" + String(recordLocation.count))
        if recordLocation.count == Constants.BATCH_SIZE {
            locQueue.append(recordLocation)
            recordLocation.removeAll()
        }
    }
 
    
    @objc func getDeviceId(){
        
        self.deviceName = UIDevice.current.name
    }
    
    @objc func getActiontime(){
//        let dateFormatter : DateFormatter = DateFormatter()
//        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let date = Date()
        let calender = Calendar.current
        let components = calender.dateComponents([.year,.month,.day,.hour,.minute,.second], from: date)
        
        let year = components.year
        let month = components.month
        let day = components.day
        let hour = components.hour
        let minute = components.minute
        let second = components.second
        
        let actiontime = String(year!) + "-" + String(month!) + "-" + String(day!) + " " + String(hour!)  + ":" + String(minute!) + ":" +  String(second!)
        
        currentTime = actiontime
    }
    @objc func timeformatter(time: String!) -> String{
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let date = dateFormatter.date(from: time)!
        dateFormatter.timeZone = TimeZone.current
        let dateString: String! = dateFormatter.string(from: date)
        return dateString
    }
    
    @objc func sampleLocation(){
        
        doesLocExceedMaxQueueSize()
     
   
        
        let location = locationManager.location

        
        let latitude = AWSDynamoDBAttributeValue()
        latitude?.s = String(format:"%.8f",location!.coordinate.latitude)
        
        let longtitude = AWSDynamoDBAttributeValue()
        longtitude?.s = String(format:"%.8f",location!.coordinate.longitude)
        
        let time = AWSDynamoDBAttributeValue()
        let timestamp = location?.timestamp.description
        
        time?.s = String(format:"%@", timeformatter(time: timestamp))
//        print(String(format:"%.8f %.8f  %@",location!.coordinate.latitude, location!.coordinate.longitude, timestamp!))
        let currentAction = AWSDynamoDBAttributeValue()
        currentAction?.s=self.action
        
        let deviceName = AWSDynamoDBAttributeValue()
        deviceName?.s=self.deviceName

        let write_request = AWSDynamoDBWriteRequest()
        write_request?.putRequest=AWSDynamoDBPutRequest()
        
        write_request?.putRequest?.item = [Constants.COLUMN_LATITUDE:latitude!, Constants.COLUMN_LONGTITUDE:longtitude!, Constants.COLUMN_TIMESTAMP:time!, Constants.COLUMN_ACTION:currentAction!,Constants.COLUMN_DEVICENAME: deviceName!]
        
        if recordLocation.contains(write_request!){

            return
        }
        
        recordLocation.append(write_request!)
        
        
        
    }
    
    @objc func sampleAccelerometer(){
        
        doesAccExceedMaxQueueSize()

        
        let data = self.motionManager.accelerometerData

        let x = AWSDynamoDBAttributeValue()
        x?.s = String(format:"%.6f",data!.acceleration.x)

        let y = AWSDynamoDBAttributeValue()
        y?.s = String(format:"%.6f",data!.acceleration.y)

        let z = AWSDynamoDBAttributeValue()
        z?.s = String(format:"%.6f",data!.acceleration.z)

        let time = AWSDynamoDBAttributeValue()
        time?.s = "\(data!.timestamp)"
      
        let currentAction = AWSDynamoDBAttributeValue()
        currentAction?.s=self.action

        let deviceName = AWSDynamoDBAttributeValue()
        deviceName?.s=self.deviceName
        
        let actionTime = AWSDynamoDBAttributeValue()
        actionTime?.s=self.currentTime
        
        let write_request = AWSDynamoDBWriteRequest()
        write_request?.putRequest=AWSDynamoDBPutRequest()

        write_request?.putRequest?.item = [Constants.COLUMN_X:x!, Constants.COLUMN_Y:y!, Constants.COLUMN_Z:z!, Constants.COLUMN_TIMESTAMP:time!, Constants.COLUMN_ACTION:currentAction!,Constants.COLUMN_DEVICENAME: deviceName!, Constants.COLUMN_ACTIONTIME: actionTime!]
        
//        msgRetrievedCnt = msgRetrievedCnt+1
        if recordAcc.contains(write_request!){
            return
        }
        
//        msgPushInCnt = msgPushInCnt+1

        recordAcc.append(write_request!)


    }
    

     @objc
    func sendExistingData()  {

        locQueue.append(recordLocation)
        accQueue.append(recordAcc)
        
//        print("🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱")
//
//        print("Total retrieved cnt:" + String(msgRetrievedCnt))
//        print("Total pushed cnt:" + String(msgPushInCnt))
//        var msgInQueue = 0
//        for batch in accQueue {
//           msgInQueue=msgInQueue+batch.count
//        }
//        print("Total Messages in queue " + String(msgInQueue))
//        msgRetrievedCnt=0
//        msgPushInCnt=0
//        print("🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱")
        
        
        
        for batch in locQueue {
            upload(array:batch, table:Constants.TABLE_LOC)
        }

//        var accQueueLoopCnt = 0
       for batch in accQueue {
//            accQueueLoopCnt+=1
            upload(array:batch, table:Constants.TABLE_ACC)
       }

        
        
        
        
//
//
//        print("🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑")
//        var accQueueCnt = accQueue.count
//
//        print("Total msgUploadedCnt:" + String(msgUploadedCnt))
//        print("Total sets in queue:" + String(accQueueCnt))
//        print("Total loop time for acc queue:" + String(accQueueLoopCnt))
//        print("Total times of upload request cnt:" + String(uploadReqCnt))
//        print("Total times of upload cnt:" + String(uploadCnt))
//        accQueueCnt=0
//        accQueueLoopCnt=0
//        msgUploadedCnt=0
//        uploadCnt=0
//        uploadReqCnt=0
//        print("🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑🐑")
       cleanCache()
    }
    
    var msgUploadedCnt=0
    var uploadReqCnt=0
    var uploadCnt=0
    func upload(array:[AWSDynamoDBWriteRequest]!, table:String!) {
        let batchWriteItemInput = AWSDynamoDBBatchWriteItemInput()
        batchWriteItemInput?.requestItems = [ table:array]
        
//        msgUploadedCnt+=array.count
//        uploadReqCnt+=1
        db.batchWriteItem(batchWriteItemInput!).continueWith { (task:AWSTask<AWSDynamoDBBatchWriteItemOutput>) -> Any? in
            if let error = task.error {
                var icon = "🌎"
                if table == Constants.TABLE_ACC {
                    icon = "❤️"
                }
                
                print("The request failed. Error: \(error) " + icon)
                return nil
            }
            if(task.isFaulted){
                print("Meow meow meow?")
            }
            
           self.uploadCnt+=1
            print("Current uploaded cnt:"+String(self.uploadCnt))
            print("I have a feeling that it succeded 🐱")
            
            return nil
        }
    }
    
    func cleanCache(){
        accQueue.removeAll()
        locQueue.removeAll()
        recordLocation.removeAll()
        recordAcc.removeAll()
    }
    
    
}

