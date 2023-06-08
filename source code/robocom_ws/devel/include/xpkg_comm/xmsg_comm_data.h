// Generated by gencpp from file xpkg_comm/xmsg_comm_data.msg
// DO NOT EDIT!


#ifndef XPKG_COMM_MESSAGE_XMSG_COMM_DATA_H
#define XPKG_COMM_MESSAGE_XMSG_COMM_DATA_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace xpkg_comm
{
template <class ContainerAllocator>
struct xmsg_comm_data_
{
  typedef xmsg_comm_data_<ContainerAllocator> Type;

  xmsg_comm_data_()
    : id_c(0)
    , id_t(0)
    , id_n(0)
    , id_f(0)
    , len(0)
    , data()
    , time()  {
      data.assign(0);
  }
  xmsg_comm_data_(const ContainerAllocator& _alloc)
    : id_c(0)
    , id_t(0)
    , id_n(0)
    , id_f(0)
    , len(0)
    , data()
    , time()  {
  (void)_alloc;
      data.assign(0);
  }



   typedef uint8_t _id_c_type;
  _id_c_type id_c;

   typedef uint8_t _id_t_type;
  _id_t_type id_t;

   typedef uint8_t _id_n_type;
  _id_n_type id_n;

   typedef uint8_t _id_f_type;
  _id_f_type id_f;

   typedef uint8_t _len_type;
  _len_type len;

   typedef boost::array<uint8_t, 8>  _data_type;
  _data_type data;

   typedef ros::Time _time_type;
  _time_type time;





  typedef boost::shared_ptr< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> const> ConstPtr;

}; // struct xmsg_comm_data_

typedef ::xpkg_comm::xmsg_comm_data_<std::allocator<void> > xmsg_comm_data;

typedef boost::shared_ptr< ::xpkg_comm::xmsg_comm_data > xmsg_comm_dataPtr;
typedef boost::shared_ptr< ::xpkg_comm::xmsg_comm_data const> xmsg_comm_dataConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace xpkg_comm

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'xpkg_comm': ['/home/eaibot/robocom_ws/src/xpkg_comm/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
{
  static const char* value()
  {
    return "9bb3169c96726b4ca470a5d97bf0777b";
  }

  static const char* value(const ::xpkg_comm::xmsg_comm_data_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x9bb3169c96726b4cULL;
  static const uint64_t static_value2 = 0xa470a5d97bf0777bULL;
};

template<class ContainerAllocator>
struct DataType< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
{
  static const char* value()
  {
    return "xpkg_comm/xmsg_comm_data";
  }

  static const char* value(const ::xpkg_comm::xmsg_comm_data_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 id_c\n\
uint8 id_t\n\
uint8 id_n\n\
uint8 id_f\n\
uint8 len\n\
uint8[8] data\n\
time time\n\
";
  }

  static const char* value(const ::xpkg_comm::xmsg_comm_data_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.id_c);
      stream.next(m.id_t);
      stream.next(m.id_n);
      stream.next(m.id_f);
      stream.next(m.len);
      stream.next(m.data);
      stream.next(m.time);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct xmsg_comm_data_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::xpkg_comm::xmsg_comm_data_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::xpkg_comm::xmsg_comm_data_<ContainerAllocator>& v)
  {
    s << indent << "id_c: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.id_c);
    s << indent << "id_t: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.id_t);
    s << indent << "id_n: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.id_n);
    s << indent << "id_f: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.id_f);
    s << indent << "len: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.len);
    s << indent << "data[]" << std::endl;
    for (size_t i = 0; i < v.data.size(); ++i)
    {
      s << indent << "  data[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.data[i]);
    }
    s << indent << "time: ";
    Printer<ros::Time>::stream(s, indent + "  ", v.time);
  }
};

} // namespace message_operations
} // namespace ros

#endif // XPKG_COMM_MESSAGE_XMSG_COMM_DATA_H
