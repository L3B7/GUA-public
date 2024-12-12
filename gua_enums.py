from enum import IntEnum, Enum
import re

class SamplingRateEnum(IntEnum):
    FASTEST = 0
    GAME = 1
    UI = 2
    NORMAL = 3
    def __str__(self):
        return self.name
    
    @staticmethod
    def get_enum_from_string(str_value)->'SamplingRateEnum':
        for member in SamplingRateEnum:
            if member.name == str_value:
                return member
        return SamplingRateEnum.FASTEST



class SensorEnum(Enum):
    def __init__(self,value:str) -> None:
        self.mode="lines"
        self.line_shape="linear"
        if(self.name == "PRX"):
            self.mode="markers"
        if(self.name == "LGT"):
            self.mode="markers"
        super().__init__()
    LCC = 'android.sensor.linear_acceleration'
    ACC = "android.sensor.accelerometer"
    GYR = 'android.sensor.gyroscope'
    GRV = 'android.sensor.gravity'
    MAG = 'android.sensor.magnetic_field'
    ROT = 'android.sensor.rotation_vector'
    LGT = 'android.sensor.light'
    PRX = 'android.sensor.proximity'

    def dataColumn(self):
        return "sensordata."+self.name
    def timestampColumn(self):
        return "timestamp."+self.name
    
    def __intValue__(self):
        return ["LCC","ACC","GYR","GRV","MAG","ROT","LGT","PRX"].index(self.name)
    def valid():
        return [SensorEnum.ACC,SensorEnum.GYR,SensorEnum.GRV,SensorEnum.MAG,SensorEnum.ROT]
    def dim(self)->int:
        if(self.name == "LCC"):
            return 3
        if(self.name == "ACC"):
            return 3
        if(self.name == "GYR"):
            return 3
        if(self.name == "GRV"):
            return 3
        if(self.name == "MAG"):
            return 3
        if(self.name == "ROT"):
            return 5
        if(self.name == "LGT"):
            return 1
        if(self.name == "PRX"):
            return 1
        return 0
    def __lt__(self, other):
        if isinstance(other, SensorEnum):
            return self.__intValue__() < other.__intValue__()
        return NotImplemented
    def from_short_string(self, short_string: str):
        short_string_lower = short_string.lower()
        for member in SensorEnum:
            if member.name.lower() == short_string_lower:
                return member
        raise ValueError(f"No matching SensorEnum member for '{short_string}'")
class InterpolModesEnum(Enum):
            NONE="none"
            LIN="linear"
            NEAREST="nearest"
            SLIN="slinear"
            CUBIC="cubic"
            QUINTIC="quintic"
            PCHIP="pchip"
            
            def __str__(self):
                return self.name
            
            @staticmethod
            def get_enum_from_string(str_value)->'InterpolModesEnum':
                for member in InterpolModesEnum:
                    if member.name == str_value:
                        return member
                return InterpolModesEnum.NONE