//
//  Constants.swift
//  csc2228
//
//  Created by Yolanda He on 2017-10-10.
//  Copyright © 2017 csc2228. All rights reserved.
//

import Foundation

struct Constants {
    static let SWITCH_RUNNING = "running"
    static let SWITCH_WALKING = "walking"
    static let SWITCH_STILL="staying_still"
    static let SWITCH_GO_UP="going_up_stairs"
    static let SWITCH_GO_DOWN="going_down_stairs"
    
    static let BATCH_SIZE=25
    
    static let INTERVAL_SAMPLING_LOCATION = 1.0
    static let INTERVAL_SAMPLING_ACCELEROMETER = 1.0/50.0 // 50Hz
    static let INTERVAL_UPLOAD = 60.0   // 10 mins/per
    
    
    
    static let COLUMN_LATITUDE = "Latitude"
    static let COLUMN_LONGTITUDE = "Longtitude"
    static let COLUMN_TIMESTAMP = "Timestamp"
    static let COLUMN_ACTION = "Action"
    static let COLUMN_DEVICENAME = "Devicename"
    static let COLUMN_ACTIONTIME = "Actiontime"
    static let COLUMN_X="X"
    static let COLUMN_Y="Y"
    static let COLUMN_Z="Z"
    
    
    static let TABLE_ACC="Accelerometer"
    static let TABLE_LOC="Location"
    
    
    
}
